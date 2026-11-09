# import warnings

import torch
import numpy as np
from pymatgen.core import Element

import utils

MAX_ATOMIC_NUMBER = 100
seed = 100
np.random.seed(seed)
torch.manual_seed(seed)


# Crystal tensor layout: 28 atom-type channels + 90 coordinates + 3 lengths + 3 angles.
TYPE_CHANNELS = 7
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
        self.rel_bias = RelativePositionBias(coord_dim, rel_hidden_dim, trans_channels // head_dim)
        self.proj_in = torch.nn.Linear(token_len, trans_channels)
        # self.pos_embed = torch.nn.Parameter(torch.randn(num_patches, channels) * 1e-2)
        if csp:
            self.csp_embed = torch.nn.Linear(token_num * TYPE_CHANNELS, trans_channels)
            self.csp_embed.weight = torch.nn.Parameter(torch.randn(trans_channels, token_num * TYPE_CHANNELS) * 1e-2)
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

        coord_feat = x_in[:, :, self.coord_offset:self.coord_offset + self.coord_dim]
        rel_bias = self.rel_bias(coord_feat)

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
        coord_full = torch.zeros(x.size(0), x.size(1), self.coord_dim, device=x.device, dtype=x.dtype)
        coord_full[:, : i + 1] = x[:, : i + 1, self.coord_offset:self.coord_offset + self.coord_dim]
        rel_bias = self.rel_bias(coord_full)[:, :, i : i + 1, : i + 1]

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

