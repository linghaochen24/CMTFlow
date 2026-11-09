# Python >= 3.10
import argparse
from argparse import BooleanOptionalAction
from pathlib import Path

import numpy as np
import torch
import torch.amp
from torch.utils.data import DataLoader

import cmtflow_s2
import utils

TYPE_CHANNELS = 7                        # 类型块通道数（原子序数的二进制位，取值 ±1）
TOTAL_CHANNELS = TYPE_CHANNELS + 96      # 7(类型二进制) + 90(坐标 one-hot) + 3(长度) + 3(角度)
TYPE_SLICE = (0, TYPE_CHANNELS)          # 类型块切片 (start, end)
COORD_SLICE = (TYPE_CHANNELS, TYPE_CHANNELS + 90)        # 坐标 one-hot 切片
LENGTH_SLICE = (TYPE_CHANNELS + 90, TYPE_CHANNELS + 93)  # 晶格长度切片
ANGLE_SLICE = (TYPE_CHANNELS + 93, TYPE_CHANNELS + 96)   # 晶格角度切片


def decode_atomic_numbers(tensor: torch.Tensor, at_mask: torch.Tensor,
                          type_table) -> torch.Tensor:
    """(B, TYPE_CHANNELS, N) 的类型二进制块 -> (B, N) 原子序数；mask 外的位置置 0。"""
    tensor = tensor.detach().float().cpu()
    at_mask = at_mask.detach().bool().cpu()
    type_bits = ((tensor.transpose(1, 2) + 1.0) / 2.0)   # (B, N, TYPE_CHANNELS) 归回 0/1
    numbers = type_table.decode_types(type_bits.reshape(-1, type_bits.size(-1)))
    numbers = numbers.reshape(type_bits.size(0), type_bits.size(1)).long()
    return numbers.masked_fill(~at_mask, 0)


def val_type_bits_and_at_mask(val_dataset):
    """汇总验证集的真实条件：类型二进制块 cond、(max_atom_num,) 原子 mask、真实晶体。"""
    at_masks, type_bits_all = [], []
    for index in range(len(val_dataset)):
        tensor, at_mask, y = val_dataset[index]
        at_masks.append(at_mask)
        type_bits_all.append(tensor[:TYPE_CHANNELS, :])   # 前 TYPE_CHANNELS 行 = 类型二进制块

    crystals = val_dataset.crystals

    at_masks = torch.stack(at_masks, dim=0)
    type_bits_all = torch.stack(type_bits_all, dim=0)
    return {"at_mask": at_masks,
            "type_bits": type_bits_all,   # (B, TYPE_CHANNELS, N) 条件（真实原子类型编码）
            "gt_crystals": crystals}


def dng_at_mask(dataset, n_samples, max_atom_num):
    """DNG：随机原子数 -> (n, max_atom_num) mask，前 count 位为 True。"""
    min_atoms, max_atoms = cmtflow_s2.dng_atom_range(dataset)
    counts = np.random.randint(min_atoms, max_atoms + 1, size=n_samples)
    at_mask = torch.zeros(n_samples, max_atom_num, dtype=torch.bool)
    for row, count in enumerate(counts):
        at_mask[row, : int(count)] = True
    return at_mask


def sample_and_denoise(model, z, y, cond, at_mask, b,
                       token_length, atom_nums, noise_std, enable_amp, args):
    """从先验 z 反向采样，再用一步梯度做去噪/langevin 修正，返回修正后的样本 x。

    :param cond: 条件（CSP 下为真实原子类型二进制块）；at_mask 标记真实原子位。
    :param y / b / token_length / atom_nums: 保留的旧参数位，当前实现未使用。
    """
    with torch.no_grad():
        with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
            x_sample = model.reverse(z, cond=cond, at_mask=at_mask)   # 初始采样
    x = torch.clone(x_sample).detach()
    x.requires_grad = True
    with torch.autocast(device_type=args.device, dtype=torch.bfloat16, enabled=enable_amp):
        z, _, logdets = model(x, cond=cond, at_mask=at_mask)
    flow_loss = model.get_loss(z, logdets, at_mask=at_mask)
    flow_grad = torch.autograd.grad(flow_loss, [x])[0]
    x.grad = flow_grad
    torch.nn.utils.clip_grad_norm_([x], max_norm=1.0)
    # exclude the factor of 'batch_size' parameters
    denoise_lr = (at_mask.sum(dim=1, keepdim=True) * noise_std ** 2)  # (b,1)
    denoise_lr = denoise_lr.unsqueeze(-1)  # (b,1,1)
    x = (x - denoise_lr * flow_grad).detach()

    return x


def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running on a single device: {args.device}')

    utils.set_random_seed(args.seed)
    print(f'Seed has been fixed to {args.seed}.')

    enabled_amp = args.noise_type == 'gaussian'

    if args.dry_run:
        args.limit = args.limit or 64
        args.num_samples = min(args.num_samples, 64)
        args.sample_batch_size = min(args.sample_batch_size, 16)
        args.sample_freq = 1
        args.batch_size = min(args.batch_size, max(1, args.limit // 2))

    max_atom_num = int(args.dataset.split('_')[-1])
    n_gen = args.num_samples // args.sample_batch_size * args.sample_batch_size   # 每轮实际采样数

    z_samples = torch.randn(args.num_samples, TOTAL_CHANNELS - TYPE_CHANNELS * args.csp, max_atom_num, device=args.device)
    # z_samples: 先验噪声 (num_samples, flow_channels, max_atom_num)，CSP 下已去掉类型通道

    train_records = utils.get_data(args.data / args.dataset / 'train_ori.pkl')
    val_records = utils.get_data(args.data / args.dataset / 'val_ori.pkl')
    if args.limit:
        train_records, val_records = train_records[: args.limit], val_records[: args.limit]

    train_dataset = utils.CrystalTensorDataset(train_records, args.dataset, max_atom_num)
    val_dataset = utils.CrystalTensorDataset(val_records, args.dataset, max_atom_num)
    val_data = val_type_bits_and_at_mask(val_dataset)
    print(f'Train: {len(train_dataset)} | Val: {len(val_dataset)} | max_atom_num: {max_atom_num}')

    # DNG：gt 指纹只算一次；采样数为 0 时不计算
    gt_comp_fp = gt_struct_fp = None
    if not args.csp and n_gen > 0 and utils.MATMINER_AVAILABLE:
        gt_comp_fp, gt_struct_fp = utils.get_fingerprints(val_data['gt_crystals'])

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=False)

    if len(train_loader) == 0:
        raise ValueError('the training split is smaller than --batch_size; lower it or raise --limit')

    model = cmtflow_s2.Model(
        in_channels=TOTAL_CHANNELS,
        token_num=max_atom_num,
        channels=args.transformer_channels,
        num_blocks=args.transformer_blocks,
        layers_per_block=args.layers_per_block,
        nvp=args.nvp,
        csp=args.csp,
    ).to(args.device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model Parameters: {num_params / 1e6:.2f} M')

    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), lr=args.lr, weight_decay=1e-4)
    lr_schedule = utils.CosineLRSchedule(optimizer, len(train_loader), args.epochs * len(train_loader),
                                         1e-6, args.lr)
    scaler = torch.amp.GradScaler(enabled=enabled_amp)
    type_table = utils.TypeTableModule()

    args.logdir.mkdir(parents=True, exist_ok=True)
    mode_tag = 'csp' if args.csp else 'dng'
    ckpt_file = args.logdir / (
        f'{args.dataset}_cmtflow_{args.transformer_channels}_{args.transformer_blocks}'
        f'_{args.layers_per_block}_{mode_tag}_s2.pth')

    print(f'{" Config ":-^80}')
    for key, value in sorted(vars(args).items()):
        print(f'{key:32s}: {value}')

    print(f'{" Training ":-^80}')
    for epoch in range(args.epochs):
        metrics = utils.Metrics()
        current_lr = args.lr

        for tensor, at_mask, _ in train_loader:
            tensor = tensor.to(args.device)         # (B, TOTAL_CHANNELS, max_atom_num) 完整张量
            at_mask = at_mask.to(args.device)       # (B, max_atom_num) bool，标出真实原子
            cond = tensor[:, :TYPE_CHANNELS, :]     # 条件：原子类型二进制块 (B, TYPE_CHANNELS, max_atom_num)

            if args.noise_type == 'gaussian':
                tensor = tensor + args.noise_std * torch.randn_like(tensor)
            else:
                raise NotImplementedError(args.noise_type)

            optimizer.zero_grad(set_to_none=True) # signature 'set_to_none' is available at PyTorch 1.7+
            with torch.autocast(device_type=args.device, dtype=torch.bfloat16, enabled=enabled_amp):
                z, block_outputs, logdets = model(tensor, at_mask=at_mask, cond=cond)
                loss = model.get_loss(z, logdets, at_mask=at_mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(filter(lambda param: param.requires_grad, model.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
            current_lr = lr_schedule.step()

            if args.update_prior:
                model.update_prior(z.detach(), at_mask=at_mask)

            with torch.no_grad():
                masked_mse = (0.5 * z.detach().pow(2) * at_mask.unsqueeze(1)).sum() / at_mask.sum()
            metrics.update({'loss': loss, 'loss/mse(z)': masked_mse, 'loss/logdet': logdets.mean()})

            if args.dry_run:
                break

        metrics_dict = {'lr': current_lr, **metrics.compute(dist=None)}
        metrics.print(metrics_dict, epoch + 1)
        if not args.dry_run:
            print('\tLayer norm', ' '.join([f'{out.pow(2).mean():.4f}' for out in block_outputs]))

        torch.save(model.state_dict(), ckpt_file)
        print(f"The model -.py file has saved in {ckpt_file}.")

        if (epoch + 1) % args.sample_freq == 0 or args.dry_run:
            print("Sampling" + '.' * 20)
            samples_list = []
            cond = None if not args.csp else val_data['type_bits']            # 条件：真实原子类型二进制块 (B, TYPE_CHANNELS, N)
            at_mask = val_data['at_mask'] if args.csp else dng_at_mask(args.dataset, n_gen, max_atom_num)           # (B, N) bool，标出真实原子
            for i in range(n_gen // args.sample_batch_size):
                b = args.sample_batch_size
                z = z_samples[i * b: (i + 1) * b].to(args.device)          # 先验噪声 (b, flow_channels, N)
                cond_ = None if cond is None else cond[i * b: (i + 1) * b].to(args.device)
                at_mask_ = at_mask[i * b: (i + 1) * b].to(args.device)
                samples = sample_and_denoise(
                    model, z, None, cond_, at_mask_, b, 96, max_atom_num,
                    args.noise_std, enabled_amp, args
                )
                samples_list.append(samples)

            if not samples_list:   # DNG 下采样数为 0：不算指纹
                print('sampling skipped')
                continue
            all_samples = torch.cat(samples_list, dim=0)   # (B, flow_channels, N) 生成结果
            atomic_numbers = None if not args.csp else decode_atomic_numbers(cond, at_mask, type_table)
            (structures, comp_valid, struct_valid, valid,
             comp_reason, struct_reason, atomic_numbers_all) = utils.decode_tensor(all_samples, at_mask, args.dataset, type_table, atomic_numbers)

            print("comp_valid", sum(comp_valid) / len(comp_valid))
            print("struct_valid", sum(struct_valid) / len(struct_valid))
            print("valid", sum(valid) / len(valid))
            print([x for x in comp_reason if x != ''])
            print([x for x in struct_reason if x != ''])

            if args.csp:
                rec_eval = utils.RecEval(pred_crys=structures, gt_crys=val_data['gt_crystals'], pred_valid=valid)
                metrics = rec_eval.get_metrics()
            else:
                pred_comp_fp = pred_struct_fp = None
                if utils.MATMINER_AVAILABLE:
                    pred_comp_fp, pred_struct_fp = utils.get_fingerprints(structures)
                metrics = {} if not any(valid) else utils.GenEval(
                    pred_crys=structures, gt_crys=val_data['gt_crystals'],
                    pred_valid=valid, pred_comp_fp=pred_comp_fp,
                    pred_struct_fp=pred_struct_fp, gt_comp_fp=gt_comp_fp,
                    gt_struct_fp=gt_struct_fp,
                    eval_model_name=args.dataset).get_metrics()

            for k, v in metrics.items():
                print(f'{k}: {v}')

        if args.dry_run:
            print('Dry run finished.')
            break

    print('End of training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../../data/', type=Path, help='Path to data directory')
    parser.add_argument('--dataset', default='perov_5',
                        choices=['perov_5', 'carbon_24', 'mp_20', 'mpts_52'], type=str,
                        help='Only datasets whose {stage}_ori.pkl exists can be trained on.')
    parser.add_argument('--logdir', default='../../runs_cmtflow', type=Path, help='Directory to save checkpoints.')
    parser.add_argument('--limit', default=0, type=int, help='Use only the first N records (0 = all).')
    parser.add_argument('--seed', default=100, type=int, help='Random seed.')

    group_model = parser.add_argument_group('Model hyper-parameters.')
    group_model.add_argument('--transformer_channels', default=512, type=int, help='Model width')
    group_model.add_argument('--transformer_blocks', default=8, type=int, help='Number of flow blocks')
    group_model.add_argument('--layers_per_block', default=4, type=int, help='Depth per flow block')
    group_model.add_argument('--nvp', default=True, action=BooleanOptionalAction, help='Whether to use NVP')
    group_model.add_argument('--noise_std', default=0.10, type=float, help='Input noise standard deviation')
    group_model.add_argument('--noise_type', default='gaussian', choices=['gaussian'], type=str)

    group_train = parser.add_argument_group('Training hyper-parameters.')
    group_train.add_argument('--batch_size', default=256, type=int, help='Training batch size')
    group_train.add_argument('--epochs', default=100, type=int, help='Training epochs')
    group_train.add_argument('--lr', default=1e-4, type=float, help='Maximum learning rate')
    group_train.add_argument('--sample_freq', default=10, type=int, help='Validate every N epochs')
    group_train.add_argument('--update_prior', default=False, action=BooleanOptionalAction,
                             help='Also update the learned prior variance (masked).')

    group_sample = parser.add_argument_group('Sampling / validation parameters.')
    group_sample.add_argument('--num_samples', default=9045, type=int,
                              help='CSP: how many validation records to pair with a sample.')
    group_sample.add_argument('--sample_batch_size', default=1005, type=int,
                              help='Sampling batch size, i.e. how many records each pass covers.')
    group_sample.add_argument('--cfg', default=0, type=float, help='Guidance weight (0 = no guidance)')
    group_sample.add_argument('--csp', default=True, action=BooleanOptionalAction,
                             help='CSP mode: condition the flow on the target composition.')

    parser.add_argument('--dry_run', default=False, action=BooleanOptionalAction, help='Smoke test on one batch.')

    main(parser.parse_args())