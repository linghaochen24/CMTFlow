# import warnings

import torch
import numpy as np
from pymatgen.core import Element

import utils

MAX_ATOMIC_NUMBER = 100
seed = 100
np.random.seed(seed)
torch.manual_seed(seed)

# ELEMENT_FEATURE_NAMES = (
#     'X',                   # Pauling electronegativity
#     'atomic_radius',       # empirical, angstrom
#     'atomic_mass',         # amu
#     'group',               # periodic-table group
#     'row',                 # periodic-table period
#     'ionization_energy',   # first ionization energy, eV
#     'electron_affinity',   # eV
#     'mendeleev_no',        # Mendeleev number
# )
# ELEMENT_FEATURE_DIM = len(ELEMENT_FEATURE_NAMES)
#
# COMPOSITION_DIM = MAX_ATOMIC_NUMBER + ELEMENT_FEATURE_DIM # 108
#
# COMP_EMBED_DIM = 64

# Crystal tensor layout: 28 atom-type channels + 90 coordinates + 3 lengths + 3 angles.
TYPE_CHANNELS = 28
CRYSTAL_CHANNELS = 124
CSP_CHANNELS = CRYSTAL_CHANNELS - TYPE_CHANNELS # 96

# DNG mask-length range per dataset; ``None`` as the upper bound means the dataset cap.
DNG_ATOM_RANGE = {
    'perov_5': (5, 5),
    'mp_20': (1, None),
    'mpts_52': (1, None),
    'carbon_24': (6, None),
}

_ELEMENT_FEATURES: torch.Tensor | None = None


# def element_feature_table() -> torch.Tensor:
#     """Map each type of element into an embedding vector while pad number 0 will be mapped into zeros vector."""
#     global _ELEMENT_FEATURES
#     if _ELEMENT_FEATURES is None:
#         raw = torch.full((MAX_ATOMIC_NUMBER + 1, ELEMENT_FEATURE_DIM), float('nan'))
#         with warnings.catch_warnings():
#             # pymatgen warns about every missing electronegativity; the NaN it returns is exactly what the imputation below expects.
#             warnings.simplefilter('ignore')
#             for z in range(1, MAX_ATOMIC_NUMBER + 1):
#                 element = Element.from_Z(z)
#                 for index, name in enumerate(ELEMENT_FEATURE_NAMES):
#                     raw[z, index] = _element_property(element, name)
#
#         # Row 0 stays zero.
#         real = raw[1:]
#         median = torch.nanmedian(real, dim=0).values
#         if not torch.isfinite(median).all():
#             # A column that is undefined for *every* element means the attribute does not exist under that name in this pymatgen release (``X`` instead of a renamed alias, say).
#             # Imputing it would quietly feed the model a constant column, so refuse to build the table instead.
#             bad = [ELEMENT_FEATURE_NAMES[i] for i, ok in enumerate(torch.isfinite(median)) if not ok]
#             raise RuntimeError(
#                 f'pymatgen exposes no value for the element descriptors {bad}; '
#                 f'check ELEMENT_FEATURE_NAMES in __anewflow.py.')
#         real = torch.where(torch.isnan(real), median.expand_as(real), real)
#         mean = real.mean(dim=0)
#         std = real.std(dim=0).clamp(min=1e-6)
#
#         _ELEMENT_FEATURES = torch.zeros_like(raw)
#         _ELEMENT_FEATURES[1:] = (real - mean) / std
#         if not torch.isfinite(_ELEMENT_FEATURES).all():
#             raise RuntimeError('the element feature table contains NaN/inf after imputation')
#     return _ELEMENT_FEATURES


def dataset_max_atoms(dataset: str) -> int:
    """The atom cap a dataset name carries: perov_5 -> 5, mp_20 -> 20, carbon_24 -> 24, mpts_52 -> 52."""
    return int(dataset.split('_')[-1])


def dng_atom_range(dataset: str) -> tuple[int, int]:
    """The DNG mask-length range: perov_5 is fixed at 5, carbon_24 spans 6..24,
    mp_20 and mpts_52 span 1..cap."""
    low, high = DNG_ATOM_RANGE.get(dataset, (1, None))
    return low, dataset_max_atoms(dataset) if high is None else high


def _element_property(element: Element, prop: str) -> float:
    """Obtain the property(scalar) of element that u designated."""
    try:
        value = getattr(element, prop, None)
    except Exception:
        return float('nan')
    if value is None or isinstance(value, (tuple, list, str)):
        return float('nan')
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return value


def masked_apply(fn, x: torch.Tensor, at_mask: torch.Tensor | None) -> torch.Tensor:
    """Apply a function with mask."""
    if at_mask is None:
        return fn(x)

    batch_size, token_num, channels = x.shape
    flat = x.reshape(batch_size * token_num, channels)
    valid = at_mask.reshape(batch_size * token_num).bool()
    num_valid = int(valid.sum())

    if num_valid == 0:
        # Nothing is real for this batch; keep the output width without relying on empty-input support inside ``fn``.
        out = fn(flat[:1])[:0]
        return torch.zeros(batch_size, token_num, out.size(-1), dtype=out.dtype, device=out.device)

    out = fn(flat[valid])  # (num_valid, C') -- only the real tokens

    # Map every row onto a valid row (padded rows fall back to row 0) and zero the padded ones afterwards, so no padded token contributes to any gradient.
    source = torch.zeros(flat.size(0), dtype=torch.long, device=flat.device)
    source[valid] = torch.arange(num_valid, device=flat.device)
    expanded = out[source] * valid.unsqueeze(-1).to(out.dtype)
    return expanded.reshape(batch_size, token_num, out.size(-1))


def zero_pad(x: torch.Tensor, at_mask: torch.Tensor | None) -> torch.Tensor:
    """Fill with 0 where the mask is False."""
    if at_mask is None:
        return x
    shape = (at_mask.size(0), at_mask.size(1)) + (1,) * (x.dim() - 2)
    return x * at_mask.reshape(shape).to(x.dtype)


class Permutation(torch.nn.Module):
    """A base class of permutation."""
    def __init__(self, seq_length: int):
        super().__init__()
        self.seq_length = seq_length

    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        raise NotImplementedError('Overload me')


class PermutationIdentity(Permutation):
    """A child class of ``Permutation(nn.Module)``."""
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x


class PermutationFlip(Permutation):
    """A child class of ``Permutation(nn.Module)``."""
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x.flip(dims=[dim])


class RelativePositionBias(torch.nn.Module):
    def __init__(self, coord_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.coord_proj = torch.nn.Linear(coord_dim, hidden_dim)
        self.rel_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 3, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, num_heads),
        )
        self.num_heads = num_heads

    def forward(self, coord_feat: torch.Tensor) -> torch.Tensor:
        feat = self.coord_proj(coord_feat)
        B, T, H = feat.shape
        feat_i = feat.unsqueeze(2).expand(-1, -1, T, -1)
        feat_j = feat.unsqueeze(1).expand(-1, T, -1, -1)
        combined = torch.cat([feat_i, feat_j, feat_i - feat_j], dim=-1)
        bias = self.rel_mlp(combined)
        return bias.permute(0, 3, 1, 2)


class Attention(torch.nn.Module):
    """A mask attention block."""
    def __init__(self, in_channels: int, head_channels: int):
        assert in_channels % head_channels == 0
        super().__init__()
        self.norm = torch.nn.LayerNorm(in_channels)
        self.qkv = torch.nn.Linear(in_channels, in_channels * 3)
        self.proj = torch.nn.Linear(in_channels, in_channels)
        self.num_heads = in_channels // head_channels
        self.sqrt_scale = head_channels ** (-0.25)
        self.sample = False
        self.k_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}
        self.v_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        temp: float = 1.0,
        which_cache: str = 'cond',
        q_mask: torch.Tensor | None = None,
        k_mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``q_mask`` marks the valid *queries* ``(B, T)``, ``k_mask`` the valid
        *keys* ``(B, T_kv)``.  While sampling, the queries are a single step while
        the cached keys span the whole prefix, hence the two arguments."""
        B, T, C = x.size()
        x = masked_apply(lambda t: self.norm(t.float()).type(t.dtype), x, q_mask)
        qkv = masked_apply(self.qkv, x, q_mask)
        q, k, v = qkv.reshape(B, T, 3 * self.num_heads, -1).transpose(1, 2).chunk(3, dim=1)  # (b, h, t, d)

        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=2)  # note that sequence dimension is now 2
            v = torch.cat(self.v_cache[which_cache], dim=2)

        scale = self.sqrt_scale**2 / temp

        attn_mask = None
        if mask is not None or q_mask is not None or k_mask is not None or bias is not None:
            T_kv = k.size(2)
            # Built explicitly as (B, 1, T, T_kv) so it broadcasts against the head dimension of the attention scores without ever mixing the batch.
            keep = torch.ones(1, 1, T, T_kv, dtype=torch.bool, device=x.device)
            if mask is not None:
                causal = mask.bool()
                if causal.dim() == 2:
                    causal = causal.reshape(1, 1, T, T_kv)
                else:
                    causal = causal.reshape(causal.size(0), 1, T, T_kv)
                keep = keep & causal
            if q_mask is not None:
                keep = keep & q_mask.bool().reshape(q_mask.size(0), 1, T, 1)
            if k_mask is None:
                k_mask = q_mask
            if k_mask is not None:
                keep = keep & k_mask.bool().reshape(k_mask.size(0), 1, 1, T_kv)

            self_index = torch.arange(T, device=x.device) + (T_kv - T)
            fallback = torch.zeros(1, 1, T, T_kv, dtype=torch.bool, device=x.device)
            fallback[0, 0, torch.arange(T, device=x.device), self_index] = True
            keep = keep | fallback
            attn_mask = torch.zeros_like(keep, dtype=x.dtype).masked_fill(~keep, float('-inf'))
            if bias is not None:
                attn_mask = attn_mask + bias.to(attn_mask.dtype)

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale)
        x = x.transpose(1, 2).reshape(B, T, C)
        x = masked_apply(self.proj, x, q_mask)
        return x


class MLP(torch.nn.Module):
    """A normal MLP layer."""
    def __init__(self, channels: int, expansion: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(channels)
        self.main = torch.nn.Sequential(
            torch.nn.Linear(channels, channels * expansion),
            torch.nn.GELU(),
            torch.nn.Linear(channels * expansion, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(self.norm(x.float()).type(x.dtype))


class AttentionBlock(torch.nn.Module):
    """A masked attention stack layers."""
    def __init__(self, channels: int, head_channels: int, expansion: int = 4):
        super().__init__()
        self.attention = Attention(channels, head_channels)
        self.mlp = MLP(channels, expansion)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        attn_temp: float = 1.0,
        which_cache: str = 'cond',
        q_mask: torch.Tensor | None = None,
        k_mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attention(x, attn_mask, attn_temp, which_cache, q_mask=q_mask, k_mask=k_mask, bias=bias)
        x = x + masked_apply(self.mlp, x, q_mask)
        return x


class MetaBlock(torch.nn.Module):
    """A masked coupling layers."""
    attn_mask: torch.Tensor

    def __init__(
        self,
        token_len: int,
        trans_channels: int,
        token_num: int,
        permutation: Permutation,
        num_layers: int = 1,
        head_dim: int = 64,
        expansion: int = 4,
        nvp: bool = True,
        csp: bool = False,
        prop: bool = False,
        coord_dim: int = 90,
        rel_hidden_dim: int = 64,
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.coord_offset = 0 if csp else TYPE_CHANNELS
        self.rel_bias = None
        self.proj_in = torch.nn.Linear(token_len, trans_channels)
        # self.pos_embed = torch.nn.Parameter(torch.randn(num_patches, channels) * 1e-2)
        if csp:
            self.csp_embed = torch.nn.Linear(token_num * 28, trans_channels)
            self.csp_embed.weight = torch.nn.Parameter(torch.randn(trans_channels, token_num * 28) * 1e-2)
            self.csp_embed.bias = torch.nn.Parameter(torch.randn(trans_channels, ) * 1e-2)
        else:
            self.csp_embed = None

        if prop:
            self.prop_embed = torch.nn.Linear(1, trans_channels)
            self.prop_embed.weight = torch.nn.Parameter(torch.randn(trans_channels, 1) * 1e-2)
            self.prop_embed.bias = torch.nn.Parameter(torch.randn(trans_channels, ) * 1e-2)
        else:
            self.prop_embed = None

        self.attn_blocks = torch.nn.ModuleList(
            [AttentionBlock(trans_channels, head_dim, expansion) for _ in range(num_layers)]
        )
        self.nvp = nvp
        output_dim = token_len * 2 if nvp else token_len
        self.proj_out = torch.nn.Linear(trans_channels, output_dim)
        self.proj_out.weight.data.fill_(0.0)
        self.permutation = permutation
        self.register_buffer('attn_mask', torch.tril(torch.ones(token_num, token_num)))

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        at_mask: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        permuted_mask = None if at_mask is None else self.permutation(at_mask, dim=1)
        x = self.permutation(x)
        # pos_embed = self.permutation(self.pos_embed, dim=0)
        x_in = x
        x = masked_apply(self.proj_in, x, permuted_mask) # + pos_embed
        x = zero_pad(x, permuted_mask)

        if self.prop_embed is not None:
            if y is not None:
                if torch.isneginf(y).any():
                    y_mask = torch.isneginf(y).view(-1, 1)
                    y[y_mask] = y[~y_mask].mean()
                    prop_embed = self.prop_embed(y)
                else:
                    prop_embed = self.prop_embed(y)
                x = zero_pad(x + prop_embed.unsqueeze(1), permuted_mask)

        if self.csp_embed is not None:
            cond = cond.flatten(start_dim=1)
            x = zero_pad(x + self.csp_embed(cond).unsqueeze(1), permuted_mask)

        rel_bias = None

        for block in self.attn_blocks:
            x = block(x, self.attn_mask, q_mask=permuted_mask, bias=rel_bias)

        x = masked_apply(self.proj_out, x, permuted_mask)
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        # The shift would otherwise leak a valid token into the first padded slot.
        x = zero_pad(x, permuted_mask)
        x = torch.clamp(x, -1., 1.)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)

        scale = (-xa.float()).exp().type(xa.dtype)
        # Back to the original token order, so the mask has to come back too.
        z = zero_pad(self.permutation((x_in - xb) * scale, inverse=True), at_mask)

        # log|det| only over the real atoms: padded slots must not dilute it.  The
        # sum is permutation-invariant, so the permuted frame is fine here.
        if permuted_mask is None:
            logdet = -xa.mean(dim=[1, 2])
        else:
            logdet = -xa.sum(dim=[1, 2]) / permuted_mask.sum(dim=1).clamp(min=1).to(xa.dtype)
        return z, logdet

    def reverse_step(
        self,
        x: torch.Tensor,
        # pos_embed: torch.Tensor,
        i: int,
        y: torch.Tensor | None = None,
        attn_temp: float = 1.0,
        which_cache: str = 'cond',
        at_mask: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = x[:, i : i + 1]  # get i-th patch but keep the sequence dimension
        q_mask = None if at_mask is None else at_mask[:, i : i + 1]
        k_mask = None if at_mask is None else at_mask[:, : i + 1]
        rel_bias = None

        x = masked_apply(self.proj_in, x_in, q_mask) # + pos_embed[i : i + 1]
        x = zero_pad(x, q_mask)
        if self.csp_embed is not None:
            cond = cond.flatten(start_dim=1)
            x = zero_pad(x + self.csp_embed(cond).unsqueeze(1), q_mask)

        if self.prop_embed is not None and y is not None:
            x = zero_pad(x + self.prop_embed(y).unsqueeze(1), q_mask)

        for block in self.attn_blocks:
            # here we use kv caching, so no causal attn_mask -- the cached prefix is
            # masked through ``k_mask`` instead
            x = block(x, attn_temp=attn_temp, which_cache=which_cache, q_mask=q_mask, k_mask=k_mask, bias=rel_bias)
        x = masked_apply(self.proj_out, x, q_mask)
        x = torch.clamp(x, -1.0, 1.0)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)
        return xa, xb

    def set_sample_mode(self, flag: bool = True):
        for m in self.modules():
            if isinstance(m, Attention):
                m.sample = flag
                m.k_cache = {'cond': [], 'uncond': []}
                m.v_cache = {'cond': [], 'uncond': []}

    def reverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = 'ab',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
        at_mask: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.permutation(x)
        if at_mask is not None:
            at_mask = self.permutation(at_mask, dim=1)
        # pos_embed = self.permutation(self.pos_embed, dim=0)
        self.set_sample_mode(True)
        T = x.size(1)
        for i in range(x.size(1) - 1):
            # za, zb = self.reverse_step(x, pos_embed, i, y, which_cache='cond',
            #                            at_mask=at_mask, cond=cond)
            za, zb = self.reverse_step(x, i, y, which_cache='cond', at_mask=at_mask, cond=cond)
            if guidance > 0 and guide_what:
                # The unconditional branch drops ``y``; the composition condition
                # is kept, matching how guidance is defined for class conditioning.
                # za_u, zb_u = self.reverse_step(
                #     x, pos_embed, i, None, attn_temp=attn_temp, which_cache='uncond',
                #     at_mask=at_mask, cond=cond
                # )
                za_u, zb_u = self.reverse_step(x, i, None, attn_temp=attn_temp, which_cache='uncond', at_mask=at_mask, cond=cond)
                if annealed_guidance:
                    g = (i + 1) / (T - 1) * guidance
                else:
                    g = guidance
                if 'a' in guide_what:
                    za = za + g * (za - za_u)
                if 'b' in guide_what:
                    zb = zb + g * (zb - zb_u)

            scale = za[:, 0].float().exp().type(za.dtype)  # get rid of the sequence dimension
            decoded = x[:, i + 1] * scale + zb[:, 0]
            if at_mask is not None:
                # Padded slots keep their prior value and stay out of the update.
                decoded = torch.where(at_mask[:, i + 1].unsqueeze(-1), decoded, x[:, i + 1])
            x[:, i + 1] = decoded
        self.set_sample_mode(False)
        return self.permutation(x, inverse=True)


class Model(torch.nn.Module):
    VAR_LR: float = 0.1
    var: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        token_num: int,
        channels: int,
        num_blocks: int,
        layers_per_block: int,
        nvp: bool = True,
        csp: bool = False,
        prop:bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.token_num = token_num
        self.csp = csp
        self.flow_channels = in_channels - TYPE_CHANNELS if csp else in_channels
        self.ttm = utils.TypeTableModule()

        permutations = [PermutationIdentity(token_num), PermutationFlip(token_num)]

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                MetaBlock(
                    self.flow_channels,
                    channels,
                    token_num,
                    permutations[i % 2],
                    layers_per_block,
                    nvp=nvp,
                    csp=csp,
                    prop=prop,
                )
            )
        self.blocks = torch.nn.ModuleList(blocks)
        # prior for nvp mode should be all ones, but needs to be learned for the vp mode.
        self.register_buffer('var', torch.ones(self.flow_channels, token_num))

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        at_mask: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        at_mask = at_mask.bool() if at_mask is not None else None
        if self.csp:
            if cond is None:
                raise ValueError('this model was built with csp=True and needs atomic_numbers')
            if x.size(1) != self.flow_channels:
                x = x[:, TYPE_CHANNELS:]

        x = x.transpose(-1, -2)  # (B, T, F)
        x = zero_pad(x, at_mask)  # hide the padded features before anything reads them

        outputs = []
        logdets = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        for block in self.blocks:
            x, logdet = block(x, y, at_mask=at_mask, cond=cond)
            logdets = logdets + logdet
            outputs.append(x)
        x = x.transpose(-1, -2)
        return x, outputs, logdets

    def update_prior(self, z: torch.Tensor, at_mask: torch.Tensor | None = None):
        """``z`` is ``(B, flow_channels, token_num)``; ``self.var`` matches it."""
        if at_mask is None:
            z2 = (z**2).mean(dim=0)
        else:
            m = at_mask.to(z.dtype).unsqueeze(1)  # (B, 1, T)
            z2 = (z**2 * m).sum(dim=0) / m.sum(dim=0).clamp(min=1.0)  # (F, T)
        self.var.lerp_(z2.detach(), weight=self.VAR_LR)

    def get_loss(self, z: torch.Tensor, logdets: torch.Tensor, at_mask: torch.Tensor | None = None):
        """``z`` is ``(B, flow_channels, token_num)``; the MSE term is averaged over
        the real atoms only, so padding cannot shrink the loss."""
        if at_mask is None:
            return 0.5 * z.pow(2).mean() - logdets.mean()
        m = at_mask.to(z.dtype).unsqueeze(1)  # (B, 1, T)
        return 0.5 * (z.pow(2) * m).sum() / m.sum().clamp(min=1.0) - logdets.mean()

    def reverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = 'ab',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
        return_sequence: bool = False,
        at_mask: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """``x`` is ``(B, flow_channels, token_num)``: under CSP the atom types are
        given, so only the ``CSP_CHANNELS`` lattice/coordinate rows are generated."""
        if at_mask is not None:
            at_mask = at_mask.bool()
            x = x * at_mask.unsqueeze(1).to(x.dtype)  # (B, F, T) * (B, 1, T)
        if self.csp:
            if cond is None:
                raise ValueError('this model was built with csp=True and needs atomic_numbers')
        seq = [x]
        x = x * self.var.sqrt()
        x = x.transpose(-1, -2)  # (B, T, F)
        for block in reversed(self.blocks):
            x = block.reverse(x, y, guidance, guide_what, attn_temp, annealed_guidance,
                              at_mask=at_mask, cond=cond)
            seq.append(x.transpose(-1, -2))
        x = x.transpose(-1, -2)
        x = x * at_mask.unsqueeze(1).to(x.dtype) if at_mask is not None else x

        if not return_sequence:
            return x
        else:
            return x, seq

# # 用于测试代码跑不跑的通...
# def _masked_run(model: Model, x: torch.Tensor, at_mask: torch.Tensor | None,
#                 atomic_numbers: torch.Tensor | None = None):
#     """Forward + loss + backward; returns the loss, parameter grads and input grad."""
#     model.zero_grad(set_to_none=True)
#     x = x.detach().clone().requires_grad_(True)
#     z, _, logdets = model(x, at_mask=at_mask, atomic_numbers=atomic_numbers)
#     loss = model.get_loss(z, logdets, at_mask=at_mask)
#     loss.backward()
#     grads = {name: param.grad.detach().clone() for name, param in model.named_parameters()}
#     return z.detach(), loss.detach(), grads, x.grad.detach()
#
#
# def _self_test():
#     """Sanity checks for the masking contract.
#
#     1. padded activations are exactly zero at every block output;
#     2. the loss, every parameter gradient and the input gradient are unchanged when
#        the *content* of the padded region is replaced, i.e. backward contributes
#        nothing for the masked data;
#     3. without a mask the very same padded region does affect the result, proving
#        the mask is what hides it;
#     4. the flow still round-trips on the real atoms and never writes padding.
#     """
#     torch.manual_seed(0)
#     # in_channels must differ from token_num: the block outputs are (B, T, F), so if
#     # the two were equal a wrongly-shaped padding mask would broadcast silently
#     # along the feature axis instead of raising.
#     batch_size, in_channels, token_num = 4, 7, 12
#     lengths = [12, 5, 3, 1]
#
#     at_mask = torch.zeros(batch_size, token_num, dtype=torch.bool)
#     for row, length in enumerate(lengths):
#         at_mask[row, :length] = True
#     padding = ~at_mask.unsqueeze(1)
#     # The block outputs are (B, T, F) -- token axis in the middle -- so they need a
#     # mask that broadcasts along F (last axis) rather than along T.
#     padding_seq = ~at_mask.unsqueeze(-1)
#
#     model = Model(in_channels, token_num, channels=128, num_blocks=3, layers_per_block=2)
#
#     base = torch.randn(batch_size, in_channels, token_num) * at_mask.unsqueeze(1)
#     # Two different fillings of the padded region: the mask alone must hide both.
#     x_garbage = torch.where(at_mask.unsqueeze(1), base, torch.randn_like(base) * 100.0)
#     x_zeros = torch.where(at_mask.unsqueeze(1), base, torch.zeros_like(base))
#
#     # --- 1. forward: padded slots are exactly zero ---------------------------
#     model.zero_grad(set_to_none=True)
#     z, outputs, _ = model(x_garbage, at_mask=at_mask)
#     assert torch.equal(z * padding, torch.zeros_like(z)), 'padded z is not zero'
#     for index, out in enumerate(outputs):
#         assert torch.equal(out * padding_seq, torch.zeros_like(out)), f'padded hidden is not zero at block {index}'
#
#     # --- 2. padding content is invisible forward and backward ----------------
#     _, loss_garbage, grads_garbage, grad_in = _masked_run(model, x_garbage, at_mask)
#     _, loss_zeros, grads_zeros, _ = _masked_run(model, x_zeros, at_mask)
#
#     assert torch.allclose(loss_garbage, loss_zeros, atol=1e-7), (
#         f'loss depends on the padded content: {loss_garbage.item()} vs {loss_zeros.item()}')
#     for name, grad in grads_garbage.items():
#         assert torch.isfinite(grad).all(), f'non-finite gradient in {name}'
#         assert torch.allclose(grad, grads_zeros[name], atol=1e-7), (
#             f'parameter gradient of {name} depends on the padded content')
#     assert torch.equal(grad_in * padding, torch.zeros_like(grad_in)), (
#         'input gradient is non-zero on the padded data')
#
#     # --- 3. the mask is what does the hiding ---------------------------------
#     _, loss_unmasked, _, _ = _masked_run(model, x_zeros, None)
#     assert not torch.allclose(loss_unmasked, loss_zeros, atol=1e-7), (
#         'unmasked run matches the masked run: the mask has no effect')
#
#     # --- 4. round trip on the real atoms -------------------------------------
#     model.zero_grad(set_to_none=True)
#     z, _, _ = model(x_garbage, at_mask=at_mask)
#     x_rec = model.reverse(z, at_mask=at_mask)
#     error = ((x_rec - x_garbage) * at_mask.unsqueeze(1)).abs().max()
#     assert error < 1e-4, f'round trip failed on valid atoms: {error}'
#     assert torch.equal(x_rec * padding, torch.zeros_like(x_rec)), 'reverse wrote into the padding'
#
#     # Sampling must leave padded slots exactly as they were handed in.
#     prior = torch.randn_like(x_garbage)
#     sampled = model.reverse(prior, at_mask=at_mask)
#     assert torch.equal(sampled * padding, torch.zeros_like(sampled)), 'sampling wrote into the padding'
#
#     # --- 5. the output heads stay in the decoder's value range ---------------
#     # ``proj_out`` feeds a log-scale head and the lattice rows, so an unbounded
#     # output lets ``exp(-xa)`` explode and lets decoded lattice lengths go
#     # non-positive.  Both forward and reverse_step must clamp it.
#     with torch.no_grad():
#         for module in model.modules():
#             if isinstance(module, MetaBlock):
#                 module.proj_out.weight.data.normal_(0.0, 10.0)
#                 module.proj_out.bias.data.normal_(0.0, 10.0)
#     z_wild, _, logdets_wild = model(x_zeros, at_mask=at_mask)
#     assert torch.isfinite(z_wild).all(), 'an extreme proj_out produced non-finite z'
#     assert torch.isfinite(logdets_wild).all(), 'an extreme proj_out produced non-finite logdet'
#     sampled_wild = model.reverse(prior, at_mask=at_mask)
#     assert torch.isfinite(sampled_wild).all(), 'an extreme proj_out produced non-finite samples'
#
#     # --- 6. CSP composition conditioning -------------------------------------
#     # The CSP flow consumes only the 96 non-type rows, so the tensor handed in is
#     # the full 124-channel one and the model drops the type rows itself.
#     cond_model = Model(CRYSTAL_CHANNELS, token_num, channels=64, num_blocks=2,
#                        layers_per_block=2, csp=True)
#     base_full = torch.randn(batch_size, CRYSTAL_CHANNELS, token_num) * at_mask.unsqueeze(1)
#     numbers = torch.randint(1, MAX_ATOMIC_NUMBER + 1, (batch_size, token_num))
#     numbers = numbers.masked_fill(~at_mask, 0)
#
#     # 6a. the condition is the composition multiset: permuting the atoms within a
#     #     sample (and the corresponding permutation of the features) must not
#     #     change the encoded vector.
#     encoded = cond_model.composition_vector(numbers, at_mask)
#     assert encoded.shape == (batch_size, COMPOSITION_DIM), \
#         f'the composition vector is {tuple(encoded.shape)}, expected {(batch_size, COMPOSITION_DIM)}'
#     permuted_numbers = torch.zeros_like(numbers)
#     for row, length in enumerate(lengths):
#         permuted_numbers[row, :length] = numbers[row, :length][torch.randperm(length)]
#     assert torch.allclose(encoded, cond_model.composition_vector(permuted_numbers, at_mask), atol=1e-6), (
#         'the composition condition depends on the atom order')
#
#     # 6b. the condition is the normalised stoichiometry, so it does not scale with
#     #     the atom count: C2O3 and C4O6 are the same composition and must encode
#     #     identically.  This is the difference from a per-atom label sum, which
#     #     would grow with the number of atoms.
#     def _one_sample(atom_numbers: list[int]):
#         numbers_row = torch.zeros(1, token_num, dtype=torch.long)
#         numbers_row[0, :len(atom_numbers)] = torch.tensor(atom_numbers)
#         mask_row = torch.zeros(1, token_num, dtype=torch.bool)
#         mask_row[0, :len(atom_numbers)] = True
#         return numbers_row, mask_row
#
#     small = _one_sample([6, 6, 8, 8, 8])                                # C2O3
#     large = _one_sample([6, 6, 6, 6, 8, 8, 8, 8, 8, 8])                 # C4O6
#     same_composition = torch.allclose(
#         cond_model.composition_vector(*small), cond_model.composition_vector(*large), atol=1e-6)
#     assert same_composition, (
#         'the composition condition scales with the atom count instead of describing the composition')
#     # ... and a different stoichiometry must still be told apart.
#     other_composition = _one_sample([6, 8, 8])                          # CO2
#     assert not torch.allclose(
#         cond_model.composition_vector(*small), cond_model.composition_vector(*other_composition), atol=1e-6), (
#         'two different compositions encode identically')
#
#     # 6c. the fractions really are atomic fractions (they sum to one per sample).
#     fraction_sums = encoded[:, :MAX_ATOMIC_NUMBER].sum(dim=1)
#     assert torch.allclose(fraction_sums, torch.ones_like(fraction_sums), atol=1e-5), \
#         f'the composition fractions do not sum to 1: {fraction_sums.tolist()}'
#
#     # 6d. padding never reaches the condition.
#     padded_numbers = numbers.clone()
#     padded_numbers[~at_mask] = MAX_ATOMIC_NUMBER
#     assert torch.allclose(encoded, cond_model.composition_vector(padded_numbers, at_mask), atol=1e-6), (
#         'the composition condition reads the padded slots')
#
#     # 6e. the element feature table is finite, standardised and zero on the pad row.
#     table = cond_model.element_features
#     assert table.shape == (MAX_ATOMIC_NUMBER + 1, ELEMENT_FEATURE_DIM), \
#         f'the element feature table is {tuple(table.shape)}'
#     assert torch.isfinite(table).all(), 'the element feature table contains NaN/inf'
#     assert torch.equal(table[0], torch.zeros(ELEMENT_FEATURE_DIM)), 'the pad row of the feature table is not zero'
#     assert torch.allclose(table[1:].mean(dim=0), torch.zeros(ELEMENT_FEATURE_DIM), atol=1e-5), \
#         'the element features are not centred'
#     assert torch.allclose(table[1:].std(dim=0), torch.ones(ELEMENT_FEATURE_DIM), atol=1e-5), \
#         'the element features are not standardised'
#
#     # 6f. the condition actually reaches the flow: two different compositions must
#     #     give different latents for the same input.  ``proj_out`` starts at zero
#     #     (so the untrained flow is the identity), hence it is perturbed first.
#     with torch.no_grad():
#         for module in cond_model.modules():
#             if isinstance(module, MetaBlock):
#                 module.proj_out.weight.data.normal_(0.0, 0.5)
#                 module.proj_out.bias.data.normal_(0.0, 0.5)
#     z_a, _, _ = cond_model(base_full, at_mask=at_mask, atomic_numbers=numbers)
#     other = numbers.clone()
#     other[:, 0] = torch.where(other[:, 0] == 1, 2, 1)
#     z_b, _, _ = cond_model(base_full, at_mask=at_mask, atomic_numbers=other)
#     assert not torch.allclose(z_a, z_b, atol=1e-6), 'the composition condition has no effect on the flow'
#     assert z_a.shape == (batch_size, CSP_CHANNELS, token_num), \
#         f'the CSP flow produced {tuple(z_a.shape)}, expected {(batch_size, CSP_CHANNELS, token_num)}'
#
#     # 6g. only the composition matters, never which token carries which atom:
#     #     rewriting the interior of a sample with the same multiset of elements
#     #     must leave both the encoded vector and the flow unchanged.
#     reshuffled = permuted_numbers
#     reshuffled_cond = cond_model.composition_vector(reshuffled, at_mask)
#     z_c, _, _ = cond_model(base_full, at_mask=at_mask, atomic_numbers=reshuffled)
#     assert torch.allclose(z_a, z_c, atol=1e-6), 'the flow depends on the atom order, not just the composition'
#
#     # 6h. padded slots stay zero and reverse honours the condition.
#     assert torch.equal(z_a * padding, torch.zeros_like(z_a)), 'conditioned forward wrote into the padding'
#     cond_prior = torch.randn(batch_size, CSP_CHANNELS, token_num)
#     sampled_cond = cond_model.reverse(cond_prior, at_mask=at_mask, atomic_numbers=numbers)
#     assert torch.equal(sampled_cond * padding, torch.zeros_like(sampled_cond)), \
#         'conditioned sampling wrote into the padding'
#     assert torch.isfinite(sampled_cond).all(), 'conditioned sampling produced non-finite values'
#
#     # 6i. the condition reaches every block, not just the first, and it survives
#     #     the permutation: swapping the two permutations' effects must still be
#     #     reversible through the conditioned flow.  ``proj_out`` is rescaled first:
#     #     the 0.5 used in 6f deliberately drives the output into the clamp in
#     #     ``MetaBlock``, and no flow is exactly invertible through a saturated
#     #     clamp, so the round trip is measured at a moderate head scale.
#     with torch.no_grad():
#         for module in cond_model.modules():
#             if isinstance(module, MetaBlock):
#                 module.proj_out.weight.data.normal_(0.0, 0.1)
#                 module.proj_out.bias.data.normal_(0.0, 0.1)
#     z_cond, _, _ = cond_model(base_full, at_mask=at_mask, atomic_numbers=numbers)
#     round_trip = cond_model.reverse(z_cond, at_mask=at_mask, atomic_numbers=numbers)
#     cond_error = ((round_trip - base_full[:, TYPE_CHANNELS:]) * at_mask.unsqueeze(1)).abs().max()
#     assert cond_error < 1e-3, f'the conditioned flow does not round-trip: {cond_error}'
#
#     # 6j. the composition condition is a function of the element counts alone:
#     #     handing the model the same counts in a differently shaped tensor (here a
#     #     sample whose atoms sit in the *suffix* of the padded axis) gives the same
#     #     condition vector.
#     shifted = torch.zeros_like(numbers)
#     shifted_mask = torch.zeros_like(at_mask)
#     for row, length in enumerate(lengths):
#         shifted[row, token_num - length:] = numbers[row, :length]
#         shifted_mask[row, token_num - length:] = True
#     assert torch.allclose(encoded, cond_model.composition_vector(shifted, shifted_mask), atol=1e-6), (
#         'the composition condition depends on where the atoms sit, not only on the composition')
#     assert torch.allclose(reshuffled_cond, encoded, atol=1e-6), 'encoding is not stable across calls'
#
#     # 6k. asking a csp model for a conditioned pass without the condition is an error.
#     try:
#         cond_model(base, at_mask=at_mask)
#     except ValueError:
#         pass
#     else:
#         raise AssertionError('a csp model accepted a forward pass with no atomic_numbers')
#
#     # 6l. the default (csp=False) model keeps working with no condition at all.
#     plain_model = Model(in_channels, token_num, channels=64, num_blocks=2, layers_per_block=2)
#     assert plain_model.composition_dim == 0, 'csp=False built a composition condition'
#     assert plain_model.element_features is None, 'csp=False built an element feature table'
#     z_plain, _, _ = plain_model(base, at_mask=at_mask)
#     assert torch.isfinite(z_plain).all(), 'the unconditioned model produced non-finite values'
#
#     print(f'ok - padded activations zero, padded backward skipped, round-trip error {error:.2e}, '
#           f'composition condition verified ({COMPOSITION_DIM}-d)')

#
# if __name__ == '__main__':
#     _self_test()
