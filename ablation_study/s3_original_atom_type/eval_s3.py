# Python >= 3.10
import argparse
import time
from argparse import BooleanOptionalAction
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import cmtflow_s3
import utils


def test_type_block_and_at_mask(test_dataset):
    """汇总测试集的真实条件：类型块 cond、(max_atom_num,) 原子 mask、真实晶体。"""
    at_masks, type_block_all = [], []
    for index in range(len(test_dataset)):
        tensor, at_mask, y = test_dataset[index]
        at_masks.append(at_mask)
        type_block_all.append(tensor[:tensor.size(1), :])   # 前 N 行 = (N, N) 类型块
    return {"at_mask": torch.stack(at_masks, dim=0),
            "type_block": torch.stack(type_block_all, dim=0),   # (B, N, N) 条件
            "gt_crystals": test_dataset.crystals}


def sample_and_denoise(model, z, cond, at_mask, noise_std, args):
    """从先验 z 反向采样，再用一步梯度做去噪修正，返回修正后的样本 x。"""
    with torch.no_grad():
        with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
            x_sample = model.reverse(z, cond=cond, at_mask=at_mask)   # 初始采样
    x = torch.clone(x_sample).detach()
    x.requires_grad = True
    with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
        z, _, logdets = model(x, cond=cond, at_mask=at_mask)
    flow_loss = model.get_loss(z, logdets, at_mask=at_mask)
    flow_grad = torch.autograd.grad(flow_loss, [x])[0]
    x.grad = flow_grad
    torch.nn.utils.clip_grad_norm_([x], max_norm=1.0)
    # exclude the factor of 'batch_size' parameters
    denoise_lr = (at_mask.sum(dim=1, keepdim=True) * noise_std ** 2)  # (b,1)
    denoise_lr = denoise_lr.unsqueeze(-1)  # (b,1,1)
    return (x - denoise_lr * flow_grad).detach()


def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running on a single device: {args.device}')

    utils.set_random_seed(args.seed)
    max_atom_num = int(args.dataset.split('_')[-1])
    type_channels = max_atom_num
    total_channels = type_channels + utils.COORD_CHANNELS + utils.LATTICE_CHANNELS

    test_dataset = utils.CrystalTensorDataset(
        utils.get_data(args.data / args.dataset / args.test_file), args.dataset, max_atom_num)
    num_samples = args.num_samples if not args.csp else (
        len(test_dataset) if args.num_samples <= 0 else min(args.num_samples, len(test_dataset)))
    test_data = test_type_block_and_at_mask(test_dataset)
    print(f'Test: {len(test_dataset)} | max_atom_num: {max_atom_num}'
          + (f' | evaluating: {num_samples} | top_k: {args.top_k}' if args.csp
             else f' | generating: {num_samples} | dng'))

    model = cmtflow_s3.Model(
        in_channels=total_channels,
        token_num=max_atom_num,
        channels=args.transformer_channels,
        num_blocks=args.transformer_blocks,
        layers_per_block=args.layers_per_block,
        nvp=args.nvp,
        csp=args.csp,
    ).to(args.device)

    mode_tag = 'csp' if args.csp else 'dng'
    ckpt_file = args.checkpoint or (args.logdir / (
        f'{args.dataset}_cmtflow_{args.transformer_channels}_{args.transformer_blocks}'
        f'_{args.layers_per_block}_{mode_tag}_s3.pth'))
    model.load_state_dict(torch.load(ckpt_file, map_location=args.device, weights_only=False))
    model.eval()
    print(f'Loaded checkpoint from {ckpt_file}.')

    type_table = utils.TypeTableModule()
    group_size = args.top_k if args.csp else 1          # 每个组（每个组分）的样本数
    total_samples = num_samples * group_size
    z_samples = torch.randn(total_samples, total_channels - type_channels * args.csp, max_atom_num,
                            device=args.device)

    # DNG：gt 指纹只算一次；采样数为 0 时不计算
    gt_comp_fp = gt_struct_fp = None
    if not args.csp and total_samples > 0 and not args.dry_run and utils.MATMINER_AVAILABLE:
        gt_comp_fp, gt_struct_fp = utils.get_fingerprints(test_data['gt_crystals'])

    print("Sampling" + '.' * 20)
    if args.device == 'cuda':
        torch.cuda.synchronize()
    sample_start_time = time.time()
    samples_list = []
    if args.csp:
        # 条件：真实原子类型 (N, N) 序列块；每个组分重复 top_k 次
        cond = test_data['type_block'][:num_samples].repeat_interleave(args.top_k, dim=0)
        at_mask = test_data['at_mask'][:num_samples].repeat_interleave(args.top_k, dim=0)
    else:
        cond = None
        min_atoms, max_atoms = cmtflow_s3.dng_atom_range(args.dataset)
        atom_counts_per_sample = np.random.randint(min_atoms, max_atoms + 1, size=total_samples)
        at_mask = torch.zeros(total_samples, max_atom_num, dtype=torch.bool)
        for row, count in enumerate(atom_counts_per_sample):
            at_mask[row, : int(count)] = True
    for i in tqdm(range(0, total_samples, args.sample_batch_size), desc='Sampling'):
        batch_size = min(args.sample_batch_size, total_samples - i)
        samples_list.append(sample_and_denoise(
            model, z_samples[i: i + batch_size], None if cond is None else cond[i: i + batch_size].to(args.device),
            at_mask[i: i + batch_size].to(args.device), args.noise_std, args))
    if not samples_list:   # DNG 下采样数为 0：不采样、不算指纹
        print('sampling skipped: num_samples=0')
        return
    all_samples = torch.cat(samples_list, dim=0)   # (total_samples, flow_channels, max_atom_num) 生成结果
    if args.device == 'cuda':
        torch.cuda.synchronize()
    sampling_time = time.time() - sample_start_time
    # DNG 模式下原子类型是自由生成的，直接从生成结果里解码；CSP 下类型已由 cond 给出
    atomic_numbers = None if not args.csp else utils.decode_atomic_numbers(cond, at_mask, type_table)
    (structures, comp_valid, struct_valid, valid,
     comp_reason, struct_reason, _) = utils.decode_tensor(
        all_samples, at_mask, args.dataset, type_table, atomic_numbers)

    print("comp_valid", sum(comp_valid) / len(comp_valid))
    print("struct_valid", sum(struct_valid) / len(struct_valid))
    print("valid", sum(valid) / len(valid))
    print([(i, i // group_size, x) for i, x in enumerate(comp_reason) if x != ''])
    print([(i, i // group_size, x) for i, x in enumerate(struct_reason) if x != ''])

    metrics = {}
    receval_time = 0.0
    gentime = 0.0
    if args.csp and not args.dry_run:
        if args.device == 'cuda':
            torch.cuda.synchronize()
        receval_start_time = time.time()
        gt_crystals = test_data['gt_crystals'][:num_samples]
        rec_eval = utils.RecEval(
            pred_crys=structures,
            gt_crys=[gt_crystals[i // args.top_k] for i in range(len(structures))],
            pred_valid=valid)
        matched_flags, rms_dists_all = [], []
        for pred_crystal, gt_crystal, is_valid in tqdm(zip(rec_eval.preds, rec_eval.gts, rec_eval.pred_valid),
                                       total=len(rec_eval.preds), desc='RecEval'):
            rms_dist = None
            if is_valid:
                try:
                    rms_dist = rec_eval.matcher.get_rms_dist(pred_crystal, gt_crystal)
                except Exception:
                    rms_dist = None
            matched_flags.append(rms_dist is not None)
            rms_dists_all.append(None if rms_dist is None else rms_dist[0])
        receval_time = time.time() - receval_start_time
        # top_k 口径：一组（同一组分）内只要有 1 个样本匹配成功就算该组命中
        top_k_hit_flags = [any(matched_flags[i: i + args.top_k]) for i in range(0, len(matched_flags), args.top_k)]
        top_k_min_rms = [min(r for r in rms_dists_all[i: i + args.top_k] if r is not None)
                         for i in range(0, len(matched_flags), args.top_k) if any(matched_flags[i: i + args.top_k])]
        metrics.update({'top_k': args.top_k,
                        'match_rate': sum(top_k_hit_flags) / len(top_k_hit_flags),
                        'sample_match_rate': sum(matched_flags) / len(matched_flags),
                        'rms_dist': sum(top_k_min_rms) / len(top_k_min_rms) if top_k_min_rms else float('nan')})
    elif not args.dry_run:
        if args.device == 'cuda':
            torch.cuda.synchronize()
        gen_start_time = time.time()
        gt_crystals = test_data['gt_crystals']
        pred_comp_fp = pred_struct_fp = None
        if utils.MATMINER_AVAILABLE:
            # 指纹：struct = 结构指纹，comp = 组分指纹（gt 已在采样前算好）
            pred_comp_fp, pred_struct_fp = utils.get_fingerprints(structures)
        if any(valid):   # 无有效晶体时跳过，wasserstein_distance 不接受空分布
            metrics.update(utils.GenEval(pred_crys=structures, gt_crys=gt_crystals,
                                         pred_valid=valid, pred_comp_fp=pred_comp_fp,
                                         pred_struct_fp=pred_struct_fp, gt_comp_fp=gt_comp_fp,
                                         gt_struct_fp=gt_struct_fp,
                                         eval_model_name=args.dataset).get_metrics())
        gentime = time.time() - gen_start_time
    for metric_name, metric_value in metrics.items():
        print(f'{metric_name}: {metric_value}')
    print(f'sampling_time: {sampling_time:.3f} s')
    if args.csp:
        print(f'receval_time: {receval_time:.3f} s')
    else:
        print(f'gentime: {gentime:.3f} s')

    args.outdir.mkdir(parents=True, exist_ok=True)
    if mode_tag == 'csp':
        out_file = args.outdir / f'{args.dataset}_{mode_tag}_{args.top_k}_samples_s3.pt'
    else:
        out_file = args.outdir / f'{args.dataset}_{mode_tag}_samples_s3.pt'
    torch.save({'samples': all_samples, 'at_mask': at_mask, 'atomic_numbers': atomic_numbers, 'valid': valid}, out_file)
    print(f'Samples saved in {out_file}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../../data/', type=Path, help='Path to data directory')
    parser.add_argument('--dataset', default='perov_5',
                        choices=['perov_5', 'carbon_24', 'mp_20', 'mpts_52'], type=str)
    parser.add_argument('--test_file', default='test_ori.pkl', type=str, help='Test split file name.')
    parser.add_argument('--logdir', default='../../runs_cmtflow', type=Path,
                        help='Directory holding the checkpoint.')
    parser.add_argument('--checkpoint', default=None, type=Path, help='Explicit checkpoint path.')
    parser.add_argument('--outdir', default='../../runs_cmtflow', type=Path, help='Where outputs are saved.')
    parser.add_argument('--num_samples', default=10000, type=int, help='Test records to evaluate (0 = all).')
    parser.add_argument('--top_k', default=1, type=int, help='Number of samples per composition.')
    parser.add_argument('--sample_batch_size', default=1000, type=int, help='Sampling batch size')
    parser.add_argument('--seed', default=100, type=int, help='Random seed.')

    group_model = parser.add_argument_group('Model hyper-parameters (must match the checkpoint).')
    group_model.add_argument('--transformer_channels', default=512, type=int, help='Model width')
    group_model.add_argument('--transformer_blocks', default=8, type=int, help='Number of flow blocks')
    group_model.add_argument('--layers_per_block', default=4, type=int, help='Depth per flow block')
    group_model.add_argument('--nvp', default=True, action=BooleanOptionalAction, help='Whether to use NVP')
    group_model.add_argument('--noise_std', default=0.10, type=float, help='Input noise standard deviation')
    group_model.add_argument('--csp', default=True, action=BooleanOptionalAction,
                             help='CSP mode: condition each sample on the target composition. '
                                  '--no-csp means de-novo generation, evaluated with GenEval.')
    parser.add_argument('--dry_run', default=False, action=BooleanOptionalAction, help='Smoke test.')

    main(parser.parse_args())
