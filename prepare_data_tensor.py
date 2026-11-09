import argparse
from pathlib import Path

import utils


def main(args):
    for stage in ['train', 'val', 'test']:
        ori_path = Path(args.data_path) / args.dataset / f'{stage}_ori.pkl'
        if ori_path.exists():
            print(f'{ori_path!r}' + 'already exists')
        else:
            original_file_path = Path(args.data_path) / args.dataset / f'{stage}.csv'
            utils.build_crystal_and_tensor_from_file(
                data_file=original_file_path,
                save_path=ori_path,
                dataset_name=args.dataset,
                num_workers=args.num_workers,
                max_atom_num=int(eval(args.dataset.split('_')[-1]))
            )
            print(f'Stage {stage} of {args.dataset} has done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data', type=Path, help='Path for each data.')
    parser.add_argument('--dataset', default='mp_20', choices=['perov_5', 'carbon_24', 'mp_20', 'mpts_52'], help='Name of the dataset')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of parallel processes for CIF -> crystal graph. When debugging will be set to 1.')

    args = parser.parse_args()

    if __debug__:
        args.num_workers = 1
    main(args)
