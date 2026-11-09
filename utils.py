# Python >=3.10
import datetime
import itertools
import math
import os
import random

import constants
import smact
import torch
import pickle

import numpy as np
import pandas as pd
import pymatgen.core

import torch.distributed
import torch.utils.data
import torch.nn.functional as F

from collections import Counter
from pathlib import Path
from p_tqdm import p_umap
from typing import Optional, Tuple, Union

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import CrystalNN
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from smact.screening import pauling_test
from tqdm import tqdm

from pyxtal import pyxtal as pyxtal_cls

try:
    from matminer.featurizers.composition.composite import ElementProperty
    from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
    MATMINER_AVAILABLE = True
except ImportError:
    ElementProperty = CrystalNNFingerprint = None
    MATMINER_AVAILABLE = False

if MATMINER_AVAILABLE:
    CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
    CompFP = ElementProperty.from_preset('magpie')
else:
    CrystalNNFP = CompFP = None

TYPE_CHANNELS = 28       # 类型块通道数 = 周期表 row(13) + col(15) 的 one-hot，取值 ±1
TOTAL_CHANNELS = 124     # 28(类型 one-hot) + 90(坐标 one-hot) + 3(晶格长度) + 3(晶格角度)
TYPE_SLICE = (0, TYPE_CHANNELS)                          # 类型 one-hot 块切片
COORD_SLICE = (TYPE_CHANNELS, TYPE_CHANNELS + 90)        # 坐标 one-hot 块切片
LENGTH_SLICE = (TYPE_CHANNELS + 90, TYPE_CHANNELS + 93)  # 晶格长度块切片
ANGLE_SLICE = (TYPE_CHANNELS + 93, TYPE_CHANNELS + 96)   # 晶格角度块切片


class StandardScaler:

    def __init__(self, means=None, stds=None, replace_nan_token=None):
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, data):
        data = np.array(data).astype(float)
        self.means = np.nanmean(data, axis=0)
        self.stds = np.nanstd(data, axis=0)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)
        return self

    def transform(self, data):
        data = np.array(data).astype(float)
        transformed_with_nan = (data - self.means) / self.stds
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)
        return transformed_with_none

    def inverse_transform(self, data):
        data = np.array(data).astype(float)
        transformed_with_nan = data * self.stds + self.means
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)
        return transformed_with_none


CompScaler = StandardScaler(means=np.array(constants.CompScalerMeans),
                            stds=np.array(constants.CompScalerStds),
                            replace_nan_token=0.)

CrystalNN = CrystalNN(distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)


def set_random_seed(seed: int) -> None:
    """Set all pseudo-random seed into a fixed value."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_data(data_path: Path):
    """Load the '.pkl' cache used to train/test a model."""
    if not data_path.exists():
        raise FileNotFoundError(
            f'{data_path} not found. Build the cache first with:\n'
            f'    python prepare_data_tensor.py --data_path <data> --dataset <dataset>')

    with open(data_path, 'rb') as file_handle:
        records = pickle.load(file_handle)

    if not isinstance(records, list) or not records:
        raise TypeError(f'{data_path} should hold a non-empty list of dicts, got {type(records)}')
    for key in ('tensor', 'mask'):
        if key not in records[0]:
            raise KeyError(
                f'{data_path}: record 0 has no {key!r} key (keys: {sorted(records[0])}). '
                f'This file was not produced by utils.build_crystal_and_tensor_from_file.')

    print(f'Loaded {len(records)} records from {data_path}.')
    return records


def num2binary(decimal_number, num_bits, position_kind) -> list:
    """Convert decimal(10) number into a binary(2) number list.

    :param decimal_number: int, 待编码的十进制数。
    :param num_bits: int, 二进制位数，也是返回列表的长度。
    :param position_kind: str, 'row' 表示编号从 0 开始（周期表行号），
        'col' 表示编号从 1 开始（周期表列号），编码前先减 1。
    :return: list[int], 长度为 num_bits 的 0/1 列表。
    """
    bits = [0 for _ in range(num_bits)]
    if position_kind == 'row':
        bits.extend(int(bit) for bit in bin(decimal_number)[2:])
    elif position_kind == 'col':
        bits.extend(int(bit) for bit in bin(decimal_number - 1)[2:])
    else:
        raise NotImplementedError("The signature position_kind only implement 'row' and 'col'.")
    return bits[-num_bits:]


def smact_validity(species, stoichiometry, use_pauling_test: bool = True, include_alloys: bool = True) -> bool:
    """SMACT checker to check whether the chemical composition is neutral or not.

    :param species: 原子种类序列，元素符号(str)或原子序数(int)。
    :param stoichiometry: 与 species 一一对应的原子个数序列。
    """
    if not (len(species) == len(stoichiometry)):
        return False
    if not species:
        return False
    for elem in species:
        if elem > constants.MAX_ATOMIC_NUM:
            return False
    try:
        if all(isinstance(element, int) for element in species):
            elem_symbols = tuple([constants.chemical_symbols[elem] for elem in species])
        else:
            elem_symbols = tuple(species)
        space = smact.element_dictionary(elem_symbols)
        smact_elems = [space[s] for s in elem_symbols]
    except (KeyError, TypeError):
        return False
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in smact_elems]
        if all(is_metal_list):
            return True
    oxidation_states_list = [e.oxidation_states for e in smact_elems]
    if any(ox is None or not ox for ox in oxidation_states_list):
        return False
    electronegs = [e.pauling_eneg for e in smact_elems]
    if use_pauling_test and None in electronegs:
        return False
    num_combinations = np.prod([len(ox) for ox in oxidation_states_list])
    if num_combinations > 1e7:
        return False
    for ox_states in itertools.product(*oxidation_states_list):
        stoichs = [(c,) for c in stoichiometry]
        try:
            is_charge_neutral = smact.neutral_ratios(ox_states, stoichs=stoichs, threshold=np.max(stoichiometry))
            if is_charge_neutral:
                electroneg_ok = True
                if use_pauling_test:
                    electroneg_ok = pauling_test(ox_states, electronegs)
                if electroneg_ok:
                    return True
        except Exception:
            continue
    return False


def structure_validity(crystal: Optional[Structure], cutoff=0.5):
    """Structure checker that if any two atoms are too close. The threshold is 0.5 A by default."""
    if crystal is None:
        return False, 'no structure'
    try:
        distance_mat = crystal.distance_matrix
        distance_mat = distance_mat + np.diag(np.ones(distance_mat.shape[0]) * (cutoff + 10.))
        if distance_mat.min() < cutoff:
            return False, '原子间距离过小。'
        elif crystal.volume < 0.1:
            return False, '晶体体积过小。'
        else:
            return True, ''
    except Exception:
        return False, '无距离矩阵'


def filter_fps(struc_fps, comp_fps):
    """Clean the 'None' fingerprints."""
    assert len(struc_fps) == len(comp_fps)
    filtered_struc_fps, filtered_comp_fps = [], []
    for struc_fp, comp_fp in zip(struc_fps, comp_fps):
        if struc_fp is not None and comp_fp is not None:
            filtered_struc_fps.append(struc_fp)
            filtered_comp_fps.append(comp_fp)
    return filtered_struc_fps, filtered_comp_fps


def compute_cov(pred_crys_comp_fp, pred_crys_struct_fp, gt_crys_comp_fp, gt_crys_struct_fp,
                struc_cutoff, comp_cutoff, num_gen_crystals=None):
    """Metrics calculator that COV-R and COV-P.

    所有 fp 参数均为指纹：comp_fp 为组分指纹，struct_fp 为结构指纹；
    gt_/pred_ 前缀分别表示真实/生成样本。
    """
    assert len(pred_crys_struct_fp) == len(pred_crys_comp_fp)
    assert len(gt_crys_struct_fp) == len(gt_crys_comp_fp)
    if num_gen_crystals is None:
        num_gen_crystals = len(pred_crys_struct_fp)
    struc_fps, comp_fps = filter_fps(pred_crys_struct_fp, pred_crys_comp_fp)
    comp_fps = CompScaler.transform(comp_fps)
    gt_comp_fps = CompScaler.transform(gt_crys_comp_fp)
    struc_fps, gt_struc_fps = np.array(struc_fps), np.array(gt_crys_struct_fp)
    comp_fps, gt_comp_fps = np.array(comp_fps), np.array(gt_comp_fps)

    struc_pdist = cdist(struc_fps, gt_struc_fps)
    comp_pdist = cdist(comp_fps, gt_comp_fps)
    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    cov_recall = np.mean(np.logical_and(struc_recall_dist <= struc_cutoff, comp_recall_dist <= comp_cutoff))
    cov_precision = np.sum(np.logical_and(struc_precision_dist <= struc_cutoff,
                                          comp_precision_dist <= comp_cutoff)) / num_gen_crystals
    return {
        'cov_recall': cov_recall,
        'cov_precision': cov_precision,
        'amsd_recall': np.mean(struc_recall_dist),
        'amsd_precision': np.mean(struc_precision_dist),
        'amcd_recall': np.mean(comp_recall_dist),
        'amcd_precision': np.mean(comp_precision_dist),
    }


def compute_novelty_and_uniqueness(pred_comp_fp, pred_struct_fp, test_comp_fp, test_struct_fp,
                                   comp_cutoff=0.2, struc_cutoff=0.1, verbose=False):
    """S.U.N crystals."""
    valid_idx = [index for index, (struct_fp, comp_fp) in enumerate(zip(pred_struct_fp, pred_comp_fp))
                 if struct_fp is not None and comp_fp is not None]
    if not valid_idx:
        return {'novelty_rate': 0.0, 'uniqueness_rate': 0.0}
    pred_comp = np.array([pred_comp_fp[index] for index in valid_idx])
    pred_struct = np.array([pred_struct_fp[index] for index in valid_idx])
    test_comp = np.array(test_comp_fp)
    test_struct = np.array(test_struct_fp)

    pred_comp_scaled = CompScaler.transform(pred_comp)
    test_comp_scaled = CompScaler.transform(test_comp)
    comp_dist_to_test = cdist(pred_comp_scaled, test_comp_scaled)
    struct_dist_to_test = cdist(pred_struct, test_struct)
    min_comp_dist = comp_dist_to_test.min(axis=1)
    min_struct_dist = struct_dist_to_test.min(axis=1)
    is_not_novel = np.logical_and(min_comp_dist <= comp_cutoff, min_struct_dist <= struc_cutoff)
    novelty_rate = np.logical_not(is_not_novel).sum() / len(pred_comp)

    comp_dist_self = cdist(pred_comp_scaled, pred_comp_scaled)
    struct_dist_self = cdist(pred_struct, pred_struct)
    np.fill_diagonal(comp_dist_self, np.inf)
    np.fill_diagonal(struct_dist_self, np.inf)
    is_duplicate = np.logical_and(comp_dist_self.min(axis=1) <= comp_cutoff,
                                  struct_dist_self.min(axis=1) <= struc_cutoff)
    uniqueness_rate = np.logical_not(is_duplicate).sum() / len(pred_comp)
    if verbose:
        print(f"  Novelty   : {novelty_rate:.4f} ({int(np.logical_not(is_not_novel).sum())}/{len(pred_comp)})")
        print(f"  Uniqueness: {uniqueness_rate:.4f} ({int(np.logical_not(is_duplicate).sum())}/{len(pred_comp)})")
    return {'novelty_rate': novelty_rate, 'uniqueness_rate': uniqueness_rate}


def get_symmetry_info(crystal: Structure, tol: float = 0.01) -> Tuple[Structure, dict]:
    """Resolve space-group symmetry information for a crystal."""
    if pyxtal_cls is None:
        raise ImportError("pyxtal is required for get_symmetry_info.")
    analyzer = SpacegroupAnalyzer(crystal, symprec=tol)
    crystal = analyzer.get_refined_structure()
    xtal = pyxtal_cls()
    try:
        xtal.from_seed(crystal, tol=0.01)
    except Exception:
        xtal.from_seed(crystal, tol=0.0001)
    space_group = xtal.group.number
    species = []        # 每个对称等价原子的元素种类
    anchors = []        # 每个原子所指的 Wyckoff 代表原子下标
    wyckoff_ops = []    # 每个原子的仿射变换矩阵（对称操作）
    coords = []         # 每个原子的分数坐标
    for site in xtal.atom_sites:
        specie = site.specie
        anchor = len(wyckoff_ops)
        coord = site.position
        for syms in site.wp:
            species.append(specie)
            wyckoff_ops.append(syms.affine_matrix)
            coords.append(syms.operate(coord))
            anchors.append(anchor)
    anchors = np.array(anchors)
    wyckoff_ops = np.array(wyckoff_ops)
    coords = np.array(coords) % 1.
    sym_info = {
        'anchors': anchors,
        'wyckoff_ops': wyckoff_ops,
        'spacegroup': space_group,
    }
    crystal = Structure(
        lattice=Lattice.from_parameters(
            *np.array(xtal.lattice.get_para(degree=True))),
        species=species,
        coords=coords,
        coords_are_cartesian=False,
    )
    return crystal, sym_info


def build_crystal(cif_str: str, niggli: bool = True, primitive: bool = False) -> Structure:
    """Create a pymatgen.core.Structure crystals."""
    crystal = Structure.from_str(cif_str, fmt='cif')
    if primitive:
        crystal = crystal.get_primitive_structure()
    if niggli:
        crystal = crystal.get_reduced_structure()
    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    return canonical_crystal


def process_one(row, prop_list, max_atom_num: int, dataset: str,
                use_space_group: bool = False, tol: float = 0.01,
                niggli: bool = True, primitive: bool = False,) -> dict:
    """
    a partial function that serves 'preprocess' function.
    :param dataset: str, dataset name.
    :param max_atom_num: int, padding to this number.
    :param row: pandas.DataFrame, one row.
    :param prop_list: property list.
    :param use_space_group: bool,
    :param tol: float, ...
    :param niggli: ...
    :param primitive: ...
    :return: Dict
    """
    cif_str = row['cif']
    crystal = build_crystal(cif_str, niggli=niggli, primitive=primitive)

    result_dict = {}
    if use_space_group:
        crystal, sym_info = get_symmetry_info(crystal, tol=tol)
        result_dict.update(sym_info)
    else:
        result_dict['spacegroup'] = 1

    properties = {key: row[key] for key in prop_list if key in row}

    data_tensor, at_mask = create_tensor_from_structure(crystal, max_atom_num=max_atom_num, dataset=dataset)
    result_dict.update({
        'mp_id': row['material_id'],
        'crystal': crystal,
        'tensor': data_tensor,      # (TOTAL_CHANNELS, max_atom_num) 多通道张量
        'mask': at_mask,            # (max_atom_num,) bool，标记真实原子位
    })
    result_dict.update(properties)
    return result_dict


def preprocess(dataframe: pd.DataFrame, num_workers: int, prop_list: list,
               use_space_group: bool = False, tol: float = 0.01,
               niggli: bool = True, primitive: bool = False, max_atom_num: int = 20, dataset: str = 'mp_20') -> list:
    """A function that process the data columns we desired."""
    num_rows = dataframe.shape[0]

    needed_columns = ['cif', 'material_id'] + list(prop_list)
    rows = dataframe[needed_columns].to_dict('records') # 将每一行转化为单个dict，计算会比df.iloc更快

    common_args = (
        prop_list, max_atom_num, dataset, use_space_group, tol, niggli, primitive
    )
    if num_workers > 1:
        unordered_results = p_umap(
            process_one,
            rows,
            *[[arg] * num_rows for arg in common_args],
            num_cpus=num_workers,
        )
    else:
        unordered_results = [
            process_one(row, *common_args)
            for row in tqdm(rows, desc='CIF -> pymatgen.Structure...')
        ]

    mpid_to_results = {result['mp_id']: result for result in unordered_results}
    mp_ids = dataframe['material_id'].tolist()
    ordered_results = [mpid_to_results[mp_id] for mp_id in mp_ids]

    return ordered_results


def build_crystal_and_tensor_from_file(data_file, save_path, dataset_name, num_workers, max_atom_num: int):
    """
    Create structure and tensor (and mask) function.
    :param data_file: Path, .csv Windows path.
    :param save_path: Path, the same parent directory.
    :param dataset_name: str, we only implement on the 4 datasets ['perov_5','mp_20', 'carbon_24', 'mpts_52']
    :param num_workers: int, decide how many cpu core to calculate the structures parallelly.
    :return: None
    """
    df = pd.read_csv(data_file)
    df = pd.DataFrame(df)

    if dataset_name == 'perov_5':
        prop = 'heat_ref'
    elif dataset_name in ['mp_20', "mpts_52"]:
        prop = 'formation_energy_per_atom'
    elif dataset_name == "carbon_24":
        prop = 'energy_per_atom'
    else:
        raise NotImplementedError("We are not implement other dataset yet.")
    results = preprocess(df, num_workers=num_workers, prop_list=[prop,], use_space_group=False, max_atom_num=max_atom_num, dataset=dataset_name)

    with open(save_path, 'wb') as file:
        pickle.dump(results, file)


def cont_space_to_categorical(_tensor: torch.Tensor, digit: int = 3) -> torch.Tensor:
    """
    将连续空间数据转化为类别数据，此函数仅作为线性空间坐标转化为类别数据之用。
    连续空间取值范围: [0, 1]
    例子：tensor([[.000,.000, .500,],[...]])
    若digit=1，则转化为[0, 0, 5](均为类别数字)
    转化为onehot编码，则为[(1, 0, 0, 0, 0, 0, 0, 0, 0, 0,), (1, 0, 0, 0, 0, 0, 0, 0, 0, 0,), (0, 0, 0, 0, 0, 1, 0, 0, 0, 0,), ...]
    :param _tensor: (B, 3)
    :param digit: int, 决定坐标精细到第几位小数，默认取3。即令0.156248... = 0.156(clip)
    :return: categorical space, size: (B, 3, digit)
    """
    # 1. 生成 10^1, 10^2, ..., 10^digit，形状 (digit,)
    powers = 10 ** torch.arange(1, digit + 1, dtype=_tensor.dtype, device=_tensor.device)
    # 2. 扩展 _tensor 最后一维用于广播: (..., 3, 1) * (digit,) -> (..., 3, digit)
    # 3. 取整后 mod 10 得到 0~9 的数字
    digits = (_tensor.unsqueeze(-1) * powers).long() % 10
    return digits


def cate_coord_onehot(cat_tensor: torch.Tensor)-> torch.Tensor:
    """
    将上面函数生成的类别数据（坐标的十进制数字）转化为 onehot 编码。
    :param cat_tensor: (N, 3, digit) 坐标各位的数字
    :return: (N, 3 * digit * 10) 的 one-hot
    """
    _size = cat_tensor.size()
    coord_onehot = torch.cat(
        [
            F.one_hot(cat_tensor[:, :, 0], 10),
            F.one_hot(cat_tensor[:, :, 1], 10),
            F.one_hot(cat_tensor[:, :, 2], 10),
        ], dim=-1
    )

    return coord_onehot.reshape(_size[0], -1)


def create_tensor_from_structure(structure: pymatgen.core.Structure,
                                 max_atom_num: int,
                                 dataset: str,
                                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform pymatgen.core.Structure into our self-defined tensor.

    返回 (data_block, at_mask)：
    data_block 形状 (TOTAL_CHANNELS, max_atom_num)，通道依次为
    [atomic_type_onehot(28, 取值 ±1), coord_onehot(90), lengths(3), angles(3)]；
    at_mask 形状 (max_atom_num,) bool，标记真实原子位（末尾为 padding）。
    """
    tensor_size = (TOTAL_CHANNELS, max_atom_num)  # (通道数, 原子 token 数)
    original_atom_num = len(structure.atomic_numbers)
    type_table = TypeTableModule()
    atomic_numbers = torch.Tensor(structure.atomic_numbers).long()
    atomic_numbers = F.pad(atomic_numbers, (0, max_atom_num - atomic_numbers.size(0)), mode='constant', value=0)
    frac_coords = torch.Tensor(structure.frac_coords)

    lengths = torch.Tensor(structure.lattice.abc)
    angles = torch.Tensor(structure.lattice.angles) / constants.SCALED_ANGLE_COEF

    # normalizing lattice_abc to [-1., 1.]
    l_max, l_min = constants.LATTICE_LENGTH_INTERVAL[dataset][1], constants.LATTICE_LENGTH_INTERVAL[dataset][0]
    assert l_max > l_min
    lengths = ((lengths - l_min) / (l_max - l_min)) * 2. - 1.

    # 类型块：周期表 (row, col) 的 one-hot 拼接（28 通道），归一化到 ±1
    atomic_type_onehot = type_table(atomic_numbers).T * 2. - 1.  # (TYPE_CHANNELS, max_atom_num)
    frac_coords = F.pad(frac_coords, (0, 0, 0, tensor_size[1] - len(frac_coords)), mode='constant', value=0.)
    frac_coords_digits = cont_space_to_categorical(frac_coords)   # (max_atom_num, 3, 3) 十进制数字
    coord_onehot = cate_coord_onehot(frac_coords_digits)          # (max_atom_num, 90) one-hot
    coord_onehot = coord_onehot.T * 2. - 1.                       # (90, max_atom_num) 归一化到 ±1
    lengths = lengths[None, ...].T.repeat_interleave(tensor_size[1], dim=1)
    angles = angles[None, ...].T.repeat_interleave(tensor_size[1], dim=1)

    data_block = torch.cat([atomic_type_onehot, coord_onehot, lengths, angles], dim=0)

    at_mask = torch.zeros((max_atom_num,), dtype=torch.bool)
    at_mask[:original_atom_num] = True

    return data_block, at_mask


class CrystalTensorDataset(torch.utils.data.Dataset):
    """Dataset Loader. Please use '.pkl' to load data."""
    def __init__(self, records: list, dataset: str, max_atom_num: int):
        self.records = records
        self.max_atom_num = max_atom_num
        self.prop = constants.PROP_NAME[dataset]
        self.atom_counts = [int(np.asarray(record['mask']).sum()) for record in records]
        self.crystals = [record['crystal'] for record in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        tensor = record['tensor']
        at_mask = record['mask']
        y = float(record[self.prop]) if self.prop in record else float('-inf')
        y = torch.tensor(y, dtype=torch.float32)

        return tensor, at_mask, y


def decode_atomic_numbers(tensor: torch.Tensor, at_mask: torch.Tensor,
                          type_table) -> torch.Tensor:
    """Read the original tensor ``(B, TOTAL_CHANNELS, N)`` into ``(B, N)`` atomic numbers.

    从类型通道（(row, col) one-hot）解码，Out of mask will be set to 0.
    """
    tensor = tensor.detach().float().cpu()
    at_mask = at_mask.detach().bool().cpu()
    type_onehot = tensor[:, TYPE_SLICE[0]:TYPE_SLICE[1], :]      # (B, TYPE_CHANNELS, N) 类型 one-hot
    type_onehot = ((type_onehot.transpose(1, 2) + 1.0) / 2.0)    # (B, N, TYPE_CHANNELS) 归回 0/1
    numbers = type_table.decode_types(type_onehot.reshape(-1, type_onehot.size(-1)))
    numbers = numbers.reshape(type_onehot.size(0), type_onehot.size(1)).long()
    return numbers.masked_fill(~at_mask, 0)


def coord_onehot_to_cont(coord_onehot: torch.Tensor) -> torch.Tensor:
    """``(n, 90)`` one-hot fractional coords -> ``(n, 3)`` continuous coords.

    Inverse of ``utils.cont_space_to_categorical`` + ``utils.cate_coord_onehot``.
    """
    x, y, z = coord_onehot[:, 0:30], coord_onehot[:, 30:60], coord_onehot[:, 60:90]

    def digits(feature: torch.Tensor) -> torch.Tensor:
        return (torch.argmax(feature[:, 0:10], dim=-1) * 1e-1
                + torch.argmax(feature[:, 10:20], dim=-1) * 1e-2
                + torch.argmax(feature[:, 20:30], dim=-1) * 1e-3)

    return torch.stack([digits(x), digits(y), digits(z)], dim=1)


def decode_tensor(
    tensor: torch.Tensor,
    at_mask: torch.Tensor,
    dataset: str,
    type_table,
    atomic_numbers: torch.Tensor | None = None,
) -> tuple[list, list, list, list, list, list, list]:
    """Decode a ``(B, TOTAL_CHANNELS, N)`` batch into structures with validity flags.

    A CSP batch is ``(B, 96, N)`` instead: it holds no atom-type rows, so the
    given ``atomic_numbers`` ``(B, N)`` are used as the species.

    Returns ``(structures, comp_valid, struct_valid, valid, comp_reason,
    struct_reason, atomic_numbers)``; entries are ``None`` / ``False`` /
    ``''`` for samples that cannot be built.  The first four entries keep their
    historical meaning so callers that only need the flags are unaffected.
    """
    assert tensor.dim() == 3, f'bad tensor {tuple(tensor.shape)}'
    # The CSP tensor drops the TYPE_CHANNELS type rows, so every slice shifts by that offset.
    offset = TOTAL_CHANNELS - tensor.size(1)
    assert offset in (0, TYPE_SLICE[1] - TYPE_SLICE[0]), f'bad tensor {tuple(tensor.shape)}'
    if offset:
        assert atomic_numbers is not None, 'a CSP tensor carries no atom types; pass atomic_numbers'
    tensor = tensor.detach().float().cpu()
    at_mask = at_mask.detach().bool().cpu()
    given_numbers = None if atomic_numbers is None else atomic_numbers.detach().cpu().long()

    l_min, l_max = constants.LATTICE_LENGTH_INTERVAL[dataset]

    type_onehot = None if offset else tensor[:, TYPE_SLICE[0]:TYPE_SLICE[1], :]        # (B, 28, N) one-hot
    coord_onehot = tensor[:, COORD_SLICE[0] - offset:COORD_SLICE[1] - offset, :]      # (B, 90, N) one-hot
    length_rows = tensor[:, LENGTH_SLICE[0] - offset:LENGTH_SLICE[1] - offset, :]     # (B, 3, N) 晶格长度
    angle_rows = tensor[:, ANGLE_SLICE[0] - offset:ANGLE_SLICE[1] - offset, :]        # (B, 3, N) 晶格角度

    structures, comp_valid, struct_valid, valid = [], [], [], []
    comp_reason, struct_reason, atomic_numbers_all = [], [], []
    for index in range(tensor.size(0)):
        sample_mask = at_mask[index]
        num_atoms = int(sample_mask.sum())

        if num_atoms < 1:
            structures.append(None)
            comp_valid.append(False)
            struct_valid.append(False)
            valid.append(False)
            comp_reason.append('atom count < 1')
            struct_reason.append('atom count < 1')
            atomic_numbers_all.append([])
            continue

        # --- atom types: (28, n) one-hot -> (n, 28) -> atomic numbers ----------
        if type_onehot is None:
            # CSP: the types were given, not generated, so just read them off.
            atomic_numbers = [int(z) for z in given_numbers[index][sample_mask].tolist()]
        else:
            type_onehot_sample = ((type_onehot[index][:, sample_mask].T + 1.0) / 2.0)  # (n, 28) 归回 0/1
            atomic_numbers = [int(z) for z in type_table.decode_types(type_onehot_sample).tolist()]

        # --- fractional coords ----------------------------------------------
        coord_onehot_sample = ((coord_onehot[index][:, sample_mask].T + 1.0) / 2.0)  # (n, 90)
        frac_coords = coord_onehot_to_cont(coord_onehot_sample).clamp(0.0, 1.0)

        # --- lattice ---------------------------------------------------------
        lengths = ((length_rows[index][:, sample_mask].mean(dim=1) + 1.0) / 2.0) * (l_max - l_min) + l_min
        angles = angle_rows[index][:, sample_mask].mean(dim=1) * constants.SCALED_ANGLE_COEF

        # ``decode_types`` yields 0 when the type head does not decode to a real
        # element; such a sample has no trustworthy composition, so it is rejected
        # outright rather than silently dropped (which would misalign the coords).
        structure = None
        reason_comp = reason_struct = 'invalid atom type'
        if min(atomic_numbers) > 0:
            counter = Counter(atomic_numbers)
            elems = tuple(sorted(counter.keys()))
            counts = np.array([counter[elem] for elem in elems])
            counts = counts / np.gcd.reduce(counts)
            reason_comp = '' if smact_validity(elems, tuple(counts.astype('int').tolist())) \
                else f'SMACT rejected composition {elems}'

            bad_lattice = bool(
                torch.isnan(lengths).any() or torch.isnan(angles).any() or torch.isnan(frac_coords).any()
            ) or float(lengths.min()) <= 0
            if bad_lattice:
                reason_struct = 'non-positive/NaN lattice'
            else:
                try:
                    structure = Structure(
                        lattice=Lattice.from_parameters(*lengths.tolist(), *angles.tolist()),
                        species=atomic_numbers,
                        coords=frac_coords.tolist(),
                        coords_are_cartesian=False,
                    )
                    ok, detail = structure_validity(structure)
                    reason_struct = '' if ok else (detail or 'structure check failed')
                except Exception as error:
                    structure, reason_struct = None, f'build failed: {error}'

        structures.append(structure)
        comp_valid.append(not reason_comp)
        struct_valid.append(not reason_struct)
        valid.append(not reason_comp and not reason_struct)
        comp_reason.append(reason_comp)
        struct_reason.append(reason_struct)
        atomic_numbers_all.append(atomic_numbers)

    return (structures, comp_valid, struct_valid, valid,
            comp_reason, struct_reason, atomic_numbers_all)


def get_fingerprints(structures: list) -> tuple[list, list]:
    """Composition / structure fingerprints for a list of crystals.

    Ported from the original ``TensorToCrystal.get_fingerprints``.  Returns
    ``(comp_fps, struct_fps)``; an entry is ``None`` whenever its fingerprint
    cannot be built, which is how the downstream coverage metrics treat a
    crystal as unusable.  Requires matminer.
    """
    if not MATMINER_AVAILABLE:
        raise RuntimeError(
            'matminer is required for composition/structure fingerprints; '
            'install it with `pip install matminer`.')

    comp_fps, struct_fps = [], []
    for structure in tqdm(structures, desc='Getting fingerprints...', miniters=100):
        if structure is None:
            comp_fps.append(None)
            struct_fps.append(None)
            continue
        try:
            comp_fps.append(CompFP.featurize(structure.composition))
        except Exception:
            comp_fps.append(None)
            struct_fps.append(None)
            continue
        try:
            site_fps = [CrystalNNFP.featurize(structure, site) for site in range(len(structure))]
        except Exception:
            # A crystal whose site fingerprint cannot be built is unusable: drop
            # both fingerprints so the pair stays aligned.
            comp_fps[-1] = None
            struct_fps.append(None)
            continue
        struct_fps.append(np.array(site_fps).mean(axis=0))
    return comp_fps, struct_fps


class RecEval:
    """重建指标（CSP）：StructureMatcher 的匹配率 + 归一化 RMSD。"""

    def __init__(self, pred_crys, gt_crys, pred_valid, stol=0.5, angle_tol=10, ltol=0.3):
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys                # 预测晶体列表
        self.gts = gt_crys                    # 真实晶体列表
        self.pred_valid = pred_valid          # 每个预测样本是否通过有效性检查
        self.batch_size = len(self.preds)

    def get_match_rate_and_rms(self):
        def rms_dist_of_one(pred_crystal, gt_crystal, is_valid):
            """单个样本的归一化 RMSD；未匹配/无效则返回 None。"""
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(pred_crystal, gt_crystal)
                rms_dist = None if rms_dist is None else rms_dist[0]
                return rms_dist
            except Exception:
                return None

        rms_dists = []
        for index in tqdm(range(len(self.preds))):
            rms_dist = rms_dist_of_one(self.preds[index], self.gts[index], self.pred_valid[index])
            if rms_dist is not None:
                rms_dists.append(rms_dist)
        if len(rms_dists) == 0:
            return {'match_rate': 0.0, 'rms_dist': float('nan')}
        rms_dists = np.array(rms_dists)
        match_rate = len(rms_dists) / len(self.preds)   # 匹配成功比例（分母为全部预测）
        mean_rms_dist = rms_dists.mean()
        return {'match_rate': match_rate, 'rms_dist': mean_rms_dist}

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics


class GenEval:
    """生成指标（DNG）：密度/元素数 W 距离 + 覆盖率 COV + 新颖性/唯一性。

    所有 *_comp_fp / *_struct_fp 均为指纹：comp 为组分指纹，struct 为结构指纹。
    """

    def __init__(self, pred_crys, gt_crys, pred_valid, pred_comp_fp, pred_struct_fp,
                 gt_comp_fp, gt_struct_fp, n_samples=64, eval_model_name=None):
        self.pred_crys = pred_crys            # 预测晶体列表
        self.gt_crys = gt_crys                # 真实晶体列表
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name
        self.pred_valid = pred_valid
        self.pred_comp_fp = pred_comp_fp      # 预测：组分指纹
        self.pred_struct_fp = pred_struct_fp  # 预测：结构指纹
        self.gt_comp_fp = gt_comp_fp          # 真实：组分指纹
        self.gt_struct_fp = gt_struct_fp      # 真实：结构指纹
        self.valid_samples = [crystal for crystal, valid_flag in zip(self.pred_crys, self.pred_valid)
                              if valid_flag is True]   # 仅保留通过有效性检查的预测晶体

    def get_density_wdist(self):
        pred_densities = [crystal.density for crystal in self.valid_samples]
        gt_densities = [crystal.density for crystal in self.gt_crys]
        return {'wdist_density': wasserstein_distance(pred_densities, gt_densities)}

    def get_num_elem_wdist(self):
        pred_nelems = [len(set(crystal.species)) for crystal in self.valid_samples]
        gt_nelems = [len(set(crystal.species)) for crystal in self.gt_crys]
        return {'wdist_num_elems': wasserstein_distance(pred_nelems, gt_nelems)}

    def get_coverage(self):
        if not MATMINER_AVAILABLE:
            return {'cov_recall': float('nan'), 'cov_precision': float('nan'),
                    'amsd_recall': float('nan'), 'amsd_precision': float('nan'),
                    'amcd_recall': float('nan'), 'amcd_precision': float('nan')}
        cutoff_dict = constants.COV_Cutoffs[self.eval_model_name]
        return compute_cov(self.pred_comp_fp, self.pred_struct_fp, self.gt_comp_fp, self.gt_struct_fp,
                           struc_cutoff=cutoff_dict['struc'], comp_cutoff=cutoff_dict['comp'])

    def get_uniqueness(self):
        if not MATMINER_AVAILABLE:
            return {'novelty_rate': float('nan'), 'uniqueness_rate': float('nan')}
        cutoff_dict = constants.COV_Cutoffs[self.eval_model_name]
        return compute_novelty_and_uniqueness(
            self.pred_comp_fp, self.pred_struct_fp, self.gt_comp_fp, self.gt_struct_fp,
            struc_cutoff=cutoff_dict['struc'], comp_cutoff=cutoff_dict['comp'])

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_density_wdist())
        metrics.update(self.get_num_elem_wdist())
        metrics.update(self.get_coverage())
        metrics.update(self.get_uniqueness())
        return metrics


class CosineLRSchedule(torch.nn.Module):
    counter: torch.Tensor

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float, max_lr: float):
        super().__init__()
        self.register_buffer('counter', torch.zeros(()))
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.set_lr(min_lr)

    def set_lr(self, lr: float) -> float:
        if self.min_lr <= lr <= self.max_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return max(self.min_lr, min(self.max_lr, lr))

    def step(self) -> float:
        with torch.no_grad():
            counter = self.counter.add_(1).item()
        if self.counter <= self.warmup_steps:
            new_lr = self.min_lr + counter / self.warmup_steps * (self.max_lr - self.min_lr)
            return self.set_lr(new_lr)
        progress = (counter - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        new_lr = self.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (self.max_lr - self.min_lr)
        return self.set_lr(new_lr)


class Distributed:
    def __init__(self):
        if os.environ.get('MASTER_PORT'):
            self.rank = int(os.environ['RANK'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.distributed = True
            torch.distributed.init_process_group('nccl', 'env://', timeout=datetime.timedelta(minutes=10))
        else:
            self.rank, self.local_rank, self.world_size = 0, 0, 1
            self.distributed = False
        torch.cuda.set_device(self.local_rank)
        self.barrier()

    def barrier(self) -> None:
        if self.distributed:
            torch.distributed.barrier()

    def gather_concat(self, x: torch.Tensor) -> torch.Tensor:
        if not self.distributed:
            return x
        x_list = [torch.empty_like(x) for _ in range(self.world_size)]
        torch.distributed.all_gather(x_list, x)
        return torch.cat(x_list)

    def __del__(self):
        if self.distributed:
            torch.distributed.destroy_process_group()


class Metrics:
    def __init__(self):
        self.metrics: dict[str, list[float]] = {}

    def update(self, metrics: dict[str, Union[torch.Tensor, float]]):
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if key in self.metrics:
                self.metrics[key].append(value)
            else:
                self.metrics[key] = [value]

    def compute(self, dist: Optional[Distributed]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in self.metrics.items():
            value = sum(value) / len(value)
            if dist is not None:
                value = dist.gather_concat(torch.tensor(value, device='cuda').view(1)).mean().item()
            out[key] = value
        return out

    @staticmethod
    def print(metrics: dict[str, float], epoch: int):
        print(f'Epoch {epoch}  Time {datetime.datetime.now()}')
        print('\n'.join((f'\t{key:40s}: {value: .4g}' for key, value in sorted(metrics.items()))))


class TypeTableModule(torch.nn.Module):
    """Periodic-table type encoder/decoder (kept from the original utils.py)."""

    def __init__(self):
        super().__init__()
        self.reordered_table = [
            ['H', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'He', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', ],
            ['Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', ],
            ['Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', ],
            ['K', 'Ca', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', ],
            ['Rb', 'Sr', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', ],
            ['Cs', 'Ba', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', ],
            ['Fr', 'Ra', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', ],
            ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', ],
            ['Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', ],
            ['Xx', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', ],
            ['Xx', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', ],
            ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', ],
            ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', ],
        ]
        reordered_map = torch.LongTensor([
            [1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, ],
            [3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, ],
            [11, 12, 13, 14, 15, 16, 17, 18, 0, 0, 0, 0, 0, 0, 0, ],
            [19, 20, 31, 32, 33, 34, 35, 36, 0, 0, 0, 0, 0, 0, 0, ],
            [37, 38, 49, 50, 51, 52, 53, 54, 0, 0, 0, 0, 0, 0, 0, ],
            [55, 56, 81, 82, 83, 84, 85, 86, 0, 0, 0, 0, 0, 0, 0, ],
            [87, 88, 113, 114, 115, 116, 117, 118, 0, 0, 0, 0, 0, 0, 0, ],
            [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 0, 0, 0, 0, 0, ],
            [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 0, 0, 0, 0, 0, ],
            [0, 72, 73, 74, 75, 76, 77, 78, 79, 80, 0, 0, 0, 0, 0, ],
            [0, 104, 105, 106, 107, 108, 109, 110, 111, 112, 0, 0, 0, 0, 0, ],
            [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, ],
            [89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, ],
        ])
        self.register_buffer("reordered_map", reordered_map)
        reordered_indices = torch.LongTensor(
            [[0, 0], [0, 7], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7],
             [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [3, 0], [3, 1],
             [7, 0], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8], [7, 9],
             [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [4, 0], [4, 1], [8, 0], [8, 1],
             [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9], [4, 2], [4, 3],
             [4, 4], [4, 5], [4, 6], [4, 7], [5, 0], [5, 1], [11, 0], [11, 1], [11, 2], [11, 3],
             [11, 4], [11, 5], [11, 6], [11, 7], [11, 8], [11, 9], [11, 10], [11, 11], [11, 12],
             [11, 13], [11, 14], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8],
             [9, 9], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [6, 0], [6, 1], [12, 0],
             [12, 1], [12, 2], [12, 3], [12, 4], [12, 5], [12, 6], [12, 7], [12, 8], [12, 9],
             [12, 10], [12, 11], [12, 12], [12, 13], [12, 14], [10, 1], [10, 2], [10, 3], [10, 4],
             [10, 5], [10, 6], [10, 7], [10, 8], [10, 9], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6],
             [6, 7], [0, 1]],
        )
        self.register_buffer("reordered_indices", reordered_indices)
        type_mask = torch.where(self.reordered_map > 0, 1.0, 0.0)
        type_mask[0, 1] = 1.0
        self.register_buffer("type_mask", type_mask)   # (num_row, num_col) 合法 (row, col) 掩码
        self.num_row = 13
        self.num_col = 15
        self.out_channels = self.num_row + self.num_col   # 28，类型 one-hot 的通道数

    def forward(self, atomic_numbers: torch.Tensor):
        return self.encode_types(atomic_numbers)

    def encode_types(self, atomic_numbers: torch.Tensor):
        """(N,) （或 (..., N)）原子序数 -> 末维为 28 的类型 one-hot（周期表 row/col 拼接）。"""
        row_col = self.reordered_indices[atomic_numbers.long() - 1]   # (..., N, 2) 的 (row, col)
        type_onehot = torch.cat([
            F.one_hot(row_col[..., 0], self.num_row),
            F.one_hot(row_col[..., 1], self.num_col),
        ], dim=-1)
        return type_onehot  # (..., N, 28)

    def decode_types(self, type_onehot: torch.Tensor):
        """(N, 28) 类型 one-hot -> (N,) 原子序数。"""
        type_onehot = type_onehot.float()
        rows = type_onehot[:, :self.num_row]
        cols = type_onehot[:, self.num_row:]
        row_indices = torch.argmax(rows, dim=-1)
        col_indices = torch.argmax(F.softmax(cols, dim=1) * self.type_mask[row_indices], dim=-1)
        atomic_numbers = self.reordered_map[[row_indices, col_indices]]
        return atomic_numbers

    def encode_types_batch(self, atomic_numbers: torch.Tensor):
        """(B, N) 原子序数 -> (B, N, 28) 类型 one-hot。"""
        return self.encode_types(atomic_numbers)






    
