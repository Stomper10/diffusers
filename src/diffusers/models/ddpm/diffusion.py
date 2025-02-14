"Largely taken and adapted from https://github.com/lucidrains/video-diffusion-pytorch"

import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many

from rotary_embedding_torch import RotaryEmbedding

from .text import tokenize, bert_embed, BERT_MODEL_DIM
from torch.utils.data import Dataset, DataLoader
from ..vq_gan_3d.vqgan import VQGAN

import matplotlib.pyplot as plt

from ...configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin
from typing import Tuple , Optional, Union


#from ...utils.accelerate_utils import apply_forward_hook
import numpy as np
# helpers functions


def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

# relative positional bias


class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance /
                                                        max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

# small helper modules


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(
            qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)

# attention along space and time


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(
            tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
        self,
        x,
        pos_bias=None,
        focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones(
                (n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

# model


class Unet3D(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        dim: int = 32,
        cond_dim=None,
        out_dim=None,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        channels: int = 3,
        attn_heads: int = 8,
        attn_dim_head: int = 32,
        use_bert_text_cond=False,
        init_dim=None,
        init_kernel_size: int = 7,
        use_sparse_linear_attn=True,
        block_type: str = 'resnet',
        resnet_groups: int = 8
    ):
        super().__init__()
        self.channels = channels

        # temporal attention and its relative positional encoding

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        def temporal_attn(dim): return EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(
            dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=copy.deepcopy(rotary_emb))) ###

        # realistically will not be able to generate that many frames of video... yet
        self.time_rel_pos_bias = RelativePositionBias(
            heads=attn_heads, max_distance=32)

        # initial conv

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size,
                                   init_kernel_size), padding=(0, init_padding, init_padding))

        self.init_temporal_attn = Residual(
            PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning

        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim

        self.null_cond_emb = nn.Parameter(
            torch.randn(1, cond_dim)) if self.has_cond else None

        cond_dim = time_dim + int(cond_dim or 0)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type

        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(
                    dim_out, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom(
            'b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(
            PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(
                    dim_in, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        cond=None,
        null_cond_prob=0.,
        focus_present_mask=None,
        # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
        prob_focus_present=0.
    ):
        assert not (self.has_cond and not exists(cond)
                    ), 'cond must be passed in if cond_dim specified'
        batch, device = x.shape[0], x.device

        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like(
            (batch,), prob_focus_present, device=device))

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)

        x = self.init_conv(x)
        r = x.clone()

        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # classifier free guidance

        if self.has_cond:
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device=device)
            cond = torch.where(rearrange(mask, 'b -> b 1'),
                               self.null_cond_emb, cond)
            t = torch.cat((t, cond), dim=-1)

        h = []

        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(
            x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)

# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        #denoise_fn, ###
        *,
        #image_size,
        #num_frames,
        #text_use_bert_cls=False,
        #channels=3,
        timesteps=1000,
        #loss_type='l1',
        use_dynamic_thres=False,  # from the Imagen paper
        dynamic_thres_percentile=0.9,
        #vqgan_ckpt=None,
    ):
        super().__init__()
        #self.channels = channels
        #self.image_size = image_size
        #self.num_frames = num_frames
        #self.denoise_fn = denoise_fn ###

        # if vqgan_ckpt:
        #     self.vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt).cuda()
        #     self.vqgan.eval()
        # else:
        #     self.vqgan = None

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        #self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters

        #self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, unet, x, t, clip_denoised: bool, cond=None, cond_scale=1.): ###
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=unet.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale)) ###

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, unet, x, t, cond=None, cond_scale=1., clip_denoised=True): ###
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            unet, x=x, t=t, clip_denoised=clip_denoised, cond=cond, cond_scale=cond_scale) ###
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, unet, shape, cond=None, cond_scale=1.): ###
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(unet, img, torch.full( ###
                (b,), i, device=device, dtype=torch.long), cond=cond, cond_scale=cond_scale)

        return img

    @torch.inference_mode()
    def sample(self, vae, unet, image_size, num_frames, channels, cond=None, cond_scale=1., batch_size=16): ###
        device = next(unet.parameters()).device

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)

        #batch_size = cond.shape[0] if exists(cond) else batch_size
        # image_size = self.image_size
        # channels = self.channels
        # num_frames = self.num_frames
        _sample = self.p_sample_loop(
            unet, (batch_size, channels, num_frames, image_size, image_size), cond=cond, cond_scale=cond_scale) ###

        if isinstance(vae, VQGAN):
            # denormalize TODO: Remove eventually
            _sample = (((_sample + 1.0) / 2.0) * (vae.codebook.embeddings.max() - # .cpu().float()
                                                  vae.codebook.embeddings.min())) + vae.codebook.embeddings.min()

            _sample = vae.decode(_sample, quantize=True)
        else:
            unnormalize_img(_sample)

        return _sample

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    # def p_losses(self, x_start, t, cond=None, noise=None, **kwargs):
    #     b, c, f, h, w, device = *x_start.shape, x_start.device
    #     noise = default(noise, lambda: torch.randn_like(x_start))

    #     x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

    #     if is_list_str(cond):
    #         cond = bert_embed(
    #             tokenize(cond), return_cls_repr=self.text_use_bert_cls)
    #         cond = cond.to(device)

    #     x_recon = self.denoise_fn(x_noisy, t, cond=cond, **kwargs)

    #     if self.loss_type == 'l1':
    #         loss = F.l1_loss(noise, x_recon)
    #     elif self.loss_type == 'l2':
    #         loss = F.mse_loss(noise, x_recon)
    #     else:
    #         raise NotImplementedError()

    #     return loss

    # def forward(self, x, *args, **kwargs):
    #     if isinstance(self.vqgan, VQGAN):
    #         with torch.no_grad():
    #             x = self.vqgan.encode(
    #                 x, quantize=False, include_embeddings=True)
    #             # normalize to -1 and 1
    #             x = ((x - self.vqgan.codebook.embeddings.min()) /
    #                  (self.vqgan.codebook.embeddings.max() -
    #                   self.vqgan.codebook.embeddings.min())) * 2.0 - 1.0
    #     else:
    #         print("Hi")
    #         x = normalize_img(x)

    #     b, device, img_size, = x.shape[0], x.device, self.image_size
    #     check_shape(x, 'b c f h w', c=self.channels,
    #                 f=self.num_frames, h=img_size, w=img_size)
    #     t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

    #     return self.p_losses(x, t, *args, **kwargs)

# DDIM
def select_timesteps(alphas_cumprod, num_inference_steps):
    """
    Given the full training schedule alphas_cumprod (a tensor of shape [T]),
    compute the SNR for each timestep as SNR = alphas_cumprod / (1 - alphas_cumprod).
    Then, compute the cumulative sum of SNR and select timesteps whose cumulative SNR
    values are evenly spaced between 0 and the total SNR.
    """
    # Compute SNR for each timestep
    snr = alphas_cumprod / (1 - alphas_cumprod)
    # Compute cumulative SNR (as a proxy for "importance" over time)
    snr_cumsum = torch.cumsum(snr, dim=0)
    total_snr = snr_cumsum[-1]
    
    # Create evenly spaced quantiles along the integrated SNR curve
    quantiles = torch.linspace(0, total_snr, steps=num_inference_steps)
    selected_steps = []
    for q in quantiles:
        # Find the first timestep where the cumulative SNR exceeds q
        idx = (snr_cumsum >= q).nonzero(as_tuple=True)[0][0].item()
        selected_steps.append(idx)
    return selected_steps

class DDIMScheduler(nn.Module):
    def __init__(self, timesteps=1000):
        super().__init__()
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.num_timesteps = timesteps

        def register_buffer(name, val):
            return self.register_buffer(name, val.to(torch.float32))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        # Buffers for x₀ reconstruction (matching your training)
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
    
    def predict_start_from_noise(self, x_t, t, noise):
        # This is the same as in training
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)
    
    def get_inference_timesteps(self, num_inference_steps):
        """
        Select timesteps for inference based on integrated SNR.
        """
        # Use the training cumulative alphas to compute SNR
        alphas_cumprod = self.alphas_cumprod  # shape [T]
        selected = select_timesteps(alphas_cumprod, num_inference_steps)
        return selected

    @torch.no_grad()
    def p_sample(self, unet, x, t, eta=0.0, cond=None, cond_scale=1.0, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        noise_pred = unet.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale)
        x0_pred = self.predict_start_from_noise(x, t, noise_pred)
        if clip_denoised:
            x0_pred = x0_pred.clamp(-1.0, 1.0)
        if (t == 0).all():
            return x0_pred

        alpha_bar_t = extract(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x.shape)
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) *
                                 (1 - alpha_bar_t / alpha_bar_prev))
        t_prev = (t - 1).clamp(min=0)
        sqrt_alpha_bar_prev = extract(self.sqrt_alphas_cumprod, t_prev, x.shape)
        alpha_bar_prev_extracted = extract(self.alphas_cumprod, t_prev, x.shape)
        sqrt_term = torch.sqrt(torch.clamp(1 - alpha_bar_prev_extracted - sigma**2, 0.0))
        x_prev = sqrt_alpha_bar_prev * x0_pred + sqrt_term * noise_pred
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().reshape(b, *((1,) * (x.ndim - 1)))
        x_prev = x_prev + nonzero_mask * sigma * noise
        return x_prev

    @torch.no_grad()
    def p_sample_loop(self, unet, shape, num_inference_steps, eta=0.0, cond=None, cond_scale=1.0):
        device = self.betas.device
        b = shape[0]
        x = torch.randn(shape, device=device)
        # Get non-uniform timesteps based on SNR weighting
        timesteps = self.get_inference_timesteps(num_inference_steps)
        # Process in reverse order
        for i in tqdm(reversed(timesteps), desc='Non-uniform DDIM sampling'):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(unet, x, t, eta=eta, cond=cond, cond_scale=cond_scale)
        return x

    @torch.no_grad()
    def sample(self, vae, unet, image_size, channels, num_frames, batch_size=16, eta=0.0, cond=None, cond_scale=1.0):
        device = next(unet.parameters()).device
        # Process conditioning if needed
        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)
        shape = (batch_size, channels, num_frames, image_size, image_size)
        x = self.p_sample_loop(unet, shape, self.num_timesteps, eta=eta, cond=cond, cond_scale=cond_scale)
        if isinstance(vae, VQGAN):
            x = (((x + 1.0) / 2.0) * (vae.codebook.embeddings.max() - vae.codebook.embeddings.min())) + vae.codebook.embeddings.min()
            x = vae.decode(x, quantize=True)
        else:
            x = unnormalize_img(x)
        return x


# trainer class


CHANNELS_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}


def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

# tensor of shape (channels, frames, height, width) -> gif


def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min())) * 1.0
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs,
                   duration=duration, loop=loop, optimize=optimize)
    return images

# gif -> (channels, frame, height, width) tensor


def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return torch.stack(tensors, dim=1)


def identity(t, *args, **kwargs):
    return t


def normalize_img(t):
    return t * 2 - 1


def unnormalize_img(t):
    return (t + 1) * 0.5


def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))


class Dataset(data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channels=3,
        num_frames=16,
        horizontal_flip=False,
        force_num_frames=True,
        exts=['gif']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in Path(
            f'{folder}').glob(f'**/*.{ext}')]

        self.cast_num_frames_fn = partial(
            cast_num_frames, frames=num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        tensor = gif_to_tensor(path, self.channels, transform=self.transform)
        return self.cast_num_frames_fn(tensor)

# trainer class


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        cfg,
        folder=None,
        dataset=None,
        *,
        ema_decay=0.995,
        num_frames=16,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder='./results',
        num_sample_rows=1,
        max_grad_norm=None,
        num_workers=20,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames

        self.cfg = cfg
        if dataset:
            self.ds = dataset
        else:
            assert folder is not None, 'Provide a folder path to the dataset'
            self.ds = Dataset(folder, image_size,
                              channels=channels, num_frames=num_frames)
        dl = DataLoader(self.ds, batch_size=train_batch_size,
                        shuffle=True, pin_memory=True, num_workers=num_workers)

        self.len_dataloader = len(dl)
        self.dl = cycle(dl)

        print(f'found {len(self.ds)} videos as gif files at {folder}')
        assert len(
            self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, map_location=None, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1])
                              for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(
                all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        if map_location:
            data = torch.load(milestone, map_location=map_location)
        else:
            data = torch.load(milestone)

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])

    def train(
        self,
        prob_focus_present=0.,
        focus_present_mask=None,
        log_fn=noop
    ):
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl)['data'].cuda()

                with autocast(enabled=self.amp):
                    loss = self.model(
                        data,
                        prob_focus_present=prob_focus_present,
                        focus_present_mask=focus_present_mask
                    )

                    self.scaler.scale(
                        loss / self.gradient_accumulate_every).backward()

                print(f'{self.step}: {loss.item()}')

            log = {'loss': loss.item()}

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                self.ema_model.eval()

                with torch.no_grad():
                    milestone = self.step // self.save_and_sample_every
                    num_samples = self.num_sample_rows ** 2
                    batches = num_to_groups(num_samples, self.batch_size)

                    all_videos_list = list(
                        map(lambda n: self.ema_model.sample(batch_size=n), batches))
                    all_videos_list = torch.cat(all_videos_list, dim=0)

                all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))

                one_gif = rearrange(
                    all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i=self.num_sample_rows)
                video_path = str(self.results_folder / str(f'{milestone}.gif'))
                video_tensor_to_gif(one_gif, video_path)
                log = {**log, 'sample': video_path}

                # Selects one random 2D image from each 3D Image
                B, C, D, H, W = all_videos_list.shape
                frame_idx = torch.randint(0, D, [B]).cuda()
                frame_idx_selected = frame_idx.reshape(
                    -1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
                frames = torch.gather(
                    all_videos_list, 2, frame_idx_selected).squeeze(2)

                path = str(self.results_folder /
                           f'sample-{milestone}.jpg')
                plt.figure(figsize=(50, 50))
                cols = 5
                for num, frame in enumerate(frames.cpu()):
                    plt.subplot(
                        math.ceil(len(frames) / cols), cols, num + 1)
                    plt.axis('off')
                    plt.imshow(frame[0], cmap='gray')
                    plt.savefig(path)

                self.save(milestone)

            log_fn(log)
            self.step += 1

        print('training completed')



class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input, mode=self.padding_type))



class PatchUnet3D(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        dim: int = 32,
        cond_dim=None,
        out_dim=None,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        channels: int = 3,
        attn_heads: int = 8,
        attn_dim_head: int = 32,
        use_bert_text_cond=False,
        init_dim=None,
        init_kernel_size: int = 7,
        use_sparse_linear_attn=True,
        block_type: str = 'resnet',
        resnet_groups: int = 8,
        num_patch_positions: int = 27, ###
        patch_position_embedding_dim: int = 16, ###
    ):
        super().__init__()

        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        def temporal_attn(dim): return EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(
            dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=copy.deepcopy(rotary_emb))) ###

        # realistically will not be able to generate that many frames of video... yet
        self.time_rel_pos_bias = RelativePositionBias(
            heads=attn_heads, max_distance=32)

        # initial conv
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2

        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size, ###
                                   init_kernel_size), padding=(0, init_padding, init_padding))

        self.init_temporal_attn = Residual(
            PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Embedding layer for patch positions
        self.patch_position_embedding = nn.Embedding(
            num_embeddings=num_patch_positions,
            embedding_dim=patch_position_embedding_dim
        )

        # time conditioning
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning
        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim

        self.null_cond_emb = nn.Parameter(
            torch.randn(1, cond_dim)) if self.has_cond else None

        cond_dim = time_dim + int(cond_dim or 0) + patch_position_embedding_dim

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type
        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)

        # modules for all layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(
                    dim_out, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom(
            'b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(
            PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(
                    dim_in, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)

        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        patch_position, ###
        cond=None,
        null_cond_prob=0.,
        focus_present_mask=None,
        # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
        prob_focus_present=0.,
        down_block_additional_residuals=None, #####
        mid_block_additional_residual=None #####
    ):
        assert not (self.has_cond and not exists(cond)
                    ), 'cond must be passed in if cond_dim specified'
        batch, device = x.shape[0], x.device

        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like(
            (batch,), prob_focus_present, device=device))

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)

        x = self.init_conv(x)
        r = x.clone()

        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        t = self.time_mlp(time) if exists(self.time_mlp) else None
        patch_pos_emb = self.patch_position_embedding(patch_position)
        # print("### t.shape:", t.shape)
        # print("### patch_pos_emb.shape:", patch_pos_emb.shape)
        # print("### cond.shape:", cond.shape)
        # print("### guidance_emb.shape:", guidance_emb.shape)

        cond_emb_list = [t, patch_pos_emb]
        
        # classifier free guidance
        if self.has_cond:
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device=device)
            cond = torch.where(rearrange(mask, 'b -> b 1'),
                               self.null_cond_emb, cond)
            cond_emb_list.append(cond)
        
        # Concatenate all conditioning embeddings
        t = torch.cat([emb for emb in cond_emb_list if emb is not None], dim=-1)

        h = [x] ###

        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            x = downsample(x)

        # print("### len(h):", len(h))
        # print("### h.shape list:", [elem.shape for elem in h])
        # Additional residual conections for Controlnets
        if down_block_additional_residuals is not None:
            new_down_block_res_samples = []
            for down_block_res_sample, down_block_additional_residual in zip(
                h, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += [down_block_res_sample,]

            h = new_down_block_res_samples

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(
            x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t)

        # Additional residual conections for Controlnets
        if mid_block_additional_residual is not None:
            x = x + mid_block_additional_residual

        # print("### len(h):", len(h))
        # print("### h.shape list:", [elem.shape for elem in h])
        #h.pop() ###
        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1) ###
            # print("### 1 x.shape:", x.shape) # 2048
            x = block1(x, t)
            # print("### 2 x.shape:", x.shape) # 512
            x = block2(x, t)
            # print("### 3 x.shape:", x.shape) # 512
            x = spatial_attn(x)
            # print("### 4 x.shape:", x.shape) # 512
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)



class PatchGaussianDiffusion(nn.Module):
    def __init__(
        self,
        #denoise_fn, ###
        *,
        #image_size,
        #num_frames,
        #text_use_bert_cls=False,
        #channels=3,
        timesteps=1000,
        #loss_type='l1',
        use_dynamic_thres=False,  # from the Imagen paper
        dynamic_thres_percentile=0.9,
        #vqgan_ckpt=None,
    ):
        super().__init__()
        #self.channels = channels
        #self.image_size = image_size
        #self.num_frames = num_frames
        #self.denoise_fn = denoise_fn ###

        # if vqgan_ckpt:
        #     self.vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt).cuda()
        #     self.vqgan.eval()
        # else:
        #     self.vqgan = None

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        #self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters

        #self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, unet, x, t, clip_denoised: bool, patch_position, cond=None, down_block_additional_residuals=None, mid_block_additional_residual=None, cond_scale=1.): ###
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=unet.forward_with_cond_scale(x, 
                                                       t, 
                                                       patch_position, 
                                                       cond=cond, 
                                                       down_block_additional_residuals=down_block_additional_residuals,
                                                       mid_block_additional_residual=mid_block_additional_residual,
                                                       cond_scale=cond_scale)) ###

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, unet, x, t, patch_position, cond=None, down_block_additional_residuals=None, mid_block_additional_residual=None, cond_scale=1., clip_denoised=True): ###
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            unet, 
            x=x, 
            t=t, 
            clip_denoised=clip_denoised, 
            patch_position=patch_position, 
            cond=cond,
            down_block_additional_residuals=down_block_additional_residuals, 
            mid_block_additional_residual=mid_block_additional_residual, 
            cond_scale=cond_scale) ###
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, unet, controlnet, shape, patch_position, cond=None, low_res_guidance=None, cond_scale=1.): ###
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            down_block_res_samples, mid_block_res_sample = controlnet( #####
                                    x=img,
                                    time=torch.full((b,), i, device=device, dtype=torch.long), 
                                    controlnet_cond=low_res_guidance, 
                                    conditioning_scale=2.0, #####
                                    patch_position=patch_position,
                                    cond=cond
                                )
            
            img = self.p_sample(unet, 
                                img, 
                                torch.full((b,), i, device=device, dtype=torch.long), 
                                patch_position, 
                                cond=cond, 
                                down_block_additional_residuals=down_block_res_samples, 
                                mid_block_additional_residual=mid_block_res_sample, 
                                cond_scale=cond_scale)

        return img

    @torch.inference_mode()
    def sample(self, vae, unet, controlnet, image_size, num_frames, channels, patch_position, cond=None, low_res_guidance=None, cond_scale=1., batch_size=16): ###
        device = next(unet.parameters()).device

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)

        #batch_size = cond.shape[0] if exists(cond) else batch_size
        # image_size = self.image_size
        # channels = self.channels
        # num_frames = self.num_frames
        _sample = self.p_sample_loop(
            unet, 
            controlnet,
            (batch_size, channels, num_frames, image_size, image_size), 
            patch_position, 
            cond=cond, 
            low_res_guidance=low_res_guidance,
            cond_scale=cond_scale) ###

        if isinstance(vae, VQGAN):
            # denormalize TODO: Remove eventually
            _sample = (((_sample + 1.0) / 2.0) * (vae.codebook.embeddings.max() -
                                                  vae.codebook.embeddings.min())) + vae.codebook.embeddings.min()

            _sample = vae.decode(_sample, quantize=True)
        else:
            unnormalize_img(_sample)

        return _sample

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )
    

#######

class PatchControlNet(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        dim: int = 32,
        cond_dim=None,
        out_dim=None,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        channels: int = 3,
        attn_heads: int = 8,
        attn_dim_head: int = 32,
        use_bert_text_cond=False,
        init_dim=None,
        init_kernel_size: int = 7,
        use_sparse_linear_attn=True,
        block_type: str = 'resnet',
        resnet_groups: int = 8,
        num_patch_positions: int = 27, ###
        patch_position_embedding_dim: int = 16, ###
    ):
        super().__init__()

        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        def temporal_attn(dim): return EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(
            dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=copy.deepcopy(rotary_emb))) ###

        # realistically will not be able to generate that many frames of video... yet
        self.time_rel_pos_bias = RelativePositionBias(
            heads=attn_heads, max_distance=32)

        # initial conv
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2

        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size, ###
                                   init_kernel_size), padding=(0, init_padding, init_padding))

        self.init_temporal_attn = Residual(
            PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Embedding layer for patch positions
        self.patch_position_embedding = nn.Embedding(
            num_embeddings=num_patch_positions,
            embedding_dim=patch_position_embedding_dim
        )

        # time conditioning
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning
        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim

        self.null_cond_emb = nn.Parameter(
            torch.randn(1, cond_dim)) if self.has_cond else None

        cond_dim = time_dim + int(cond_dim or 0) + patch_position_embedding_dim

        # control net conditioning embedding ###
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding_VQGANEncoder(
            # embedding_dim=8,
            # n_codes=16384,
            n_hiddens=16,
            downsample=(2,2,2),
            image_channels=1,
            # restart_thres=1.0,
            # no_random_restart=False,
            norm_type="group",
            padding_type="replicate",
            num_groups=32,
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        self.controlnet_down_blocks = nn.ModuleList([]) ###

        num_resolutions = len(in_out)

        # block type
        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)

        ###
        controlnet_block = nn.Conv3d(in_out[0][0], in_out[0][0], kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)
        ###

        # modules for all layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(
                    dim_out, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

            ###
            for _ in range(1):
                controlnet_block = nn.Conv3d(dim_out, dim_out, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

            # if not is_last:
            #     controlnet_block = nn.Conv3d(dim_out, dim_out, kernel_size=1)
            #     controlnet_block = zero_module(controlnet_block)
            #     self.controlnet_down_blocks.append(controlnet_block)
            ###
        
        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom(
            'b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(
            PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        ###
        # self.controlnet_mid_block = nn.ModuleList([]) ###
        # for _ in range(4):
        controlnet_block = nn.Conv3d(mid_dim, mid_dim, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_mid_block = controlnet_block
        ###

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        controlnet_cond, ###
        conditioning_scale, ###
        patch_position,
        cond=None,
        null_cond_prob=0.,
        focus_present_mask=None,
        # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
        prob_focus_present=0.
    ):
        assert not (self.has_cond and not exists(cond)
                    ), 'cond must be passed in if cond_dim specified'
        batch, device = x.shape[0], x.device

        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like(
            (batch,), prob_focus_present, device=device))

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)

        x = self.init_conv(x)
        r = x.clone()

        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        ###
        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        x += controlnet_cond
        ###

        t = self.time_mlp(time) if exists(self.time_mlp) else None
        patch_pos_emb = self.patch_position_embedding(patch_position)
        # print("### t.shape:", t.shape)
        # print("### patch_pos_emb.shape:", patch_pos_emb.shape)
        # print("### cond.shape:", cond.shape)
        # print("### guidance_emb.shape:", guidance_emb.shape)

        cond_emb_list = [t, patch_pos_emb]
        
        # classifier free guidance
        if self.has_cond:
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device=device)
            cond = torch.where(rearrange(mask, 'b -> b 1'),
                               self.null_cond_emb, cond)
            cond_emb_list.append(cond)
        
        # Concatenate all conditioning embeddings
        t = torch.cat([emb for emb in cond_emb_list if emb is not None], dim=-1)

        h = [x] ###

        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            #h.append(x)
            x = spatial_attn(x)
            #h.append(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            #h.append(x)
            x = downsample(x)
            #h.append(x)
        
        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(
            x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t)

        ###
        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(h, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples += (down_block_res_sample,)

        h = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(x)

        h = [x * conditioning_scale for x in h]
        mid_block_res_sample *= conditioning_scale

        ###

        # for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
        #     x = torch.cat((x, h.pop()), dim=1)
        #     x = block1(x, t)
        #     x = block2(x, t)
        #     x = spatial_attn(x)
        #     x = temporal_attn(x, pos_bias=time_rel_pos_bias,
        #                       focus_present_mask=focus_present_mask)
        #     x = upsample(x)

        # x = torch.cat((x, r), dim=1)
        return h, mid_block_res_sample



def silu(x):
    return x*torch.sigmoid(x)

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)

def Normalize(in_channels, norm_type='group', num_groups=32):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        # TODO Changed num_groups from 32 to 8
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group', padding_type='replicate', num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv1 = SamePadConv3d(
            in_channels, out_channels, kernel_size=3, padding_type=padding_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv2 = SamePadConv3d(
            out_channels, out_channels, kernel_size=3, padding_type=padding_type)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv3d(
                in_channels, out_channels, kernel_size=3, padding_type=padding_type)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x+h

class Encoder(nn.Module):
    def __init__(self, n_hiddens, downsample, image_channel=3, norm_type='group', padding_type='replicate', num_groups=32):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.conv_blocks = nn.ModuleList()
        max_ds = n_times_downsample.max()

        self.conv_first = SamePadConv3d(
            image_channel, n_hiddens, kernel_size=3, padding_type=padding_type)

        for i in range(max_ds):
            block = nn.Module()
            in_channels = n_hiddens * 2**i
            out_channels = n_hiddens * 2**(i+1)
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            block.down = SamePadConv3d(
                in_channels, out_channels, 4, stride=stride, padding_type=padding_type)
            block.res = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            self.conv_blocks.append(block)
            n_times_downsample -= 1

        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type, num_groups=num_groups),
            SiLU()
        )

        self.out_channels = out_channels

    def forward(self, x):
        h = self.conv_first(x)
        for block in self.conv_blocks:
            h = block.down(h)
            h = block.res(h)
        h = self.final_block(h)
        return h


class ControlNetConditioningEmbedding_VQGANEncoder(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(
        self,
        # embedding_dim: int = 256,
        # n_codes: int = 2048,
        n_hiddens: int = 240,
        downsample: Tuple[int, ...] = (4, 4, 4),
        image_channels: int = 1,
        # restart_thres: float = 1.0,
        # no_random_restart=False,
        norm_type: str = "group",
        padding_type: str = "replicate",
        num_groups: int = 32,
    ):
        super().__init__()
        # self.cfg = cfg
        # self.embedding_dim = embedding_dim
        # self.n_codes = n_codes

        self.encoder = Encoder(n_hiddens, downsample, image_channels, norm_type, padding_type, num_groups,)
        #self.decoder = Decoder(n_hiddens, downsample, image_channels, norm_type, num_groups)
        #self.enc_out_ch = self.encoder.out_channels
        #self.pre_vq_conv = SamePadConv3d(self.enc_out_ch, embedding_dim, 1, padding_type=padding_type)
        #self.post_vq_conv = SamePadConv3d(embedding_dim, self.enc_out_ch, 1)

        #self.codebook = Codebook(n_codes, embedding_dim, no_random_restart=no_random_restart, restart_thres=restart_thres)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Encoder):
            module.gradient_checkpointing = value
    
    def forward(self, x, optimizer_idx=None, log_image=False):
        # B, C, T, H, W = x.shape

        # z = self.pre_vq_conv(self.encoder(x))
        # vq_output = self.codebook(z)
        # x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']))
        z = self.encoder(x)
        return z # x_recon, vq_output


from collections.abc import Sequence
from monai.networks.blocks import Convolution

class ControlNetConditioningEmbedding(nn.Module):
    """
    Network to encode the conditioning into a latent space.
    """

    def __init__(
        self, 
        spatial_dims: int, 
        in_channels: int, 
        out_channels: int, 
        num_channels: Sequence[int] = (16, 32, 96, 256)
    ):
        super().__init__()

        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(num_channels) - 1):
            channel_in = num_channels[i]
            channel_out = num_channels[i + 1]
            self.blocks.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=channel_in,
                    out_channels=channel_in,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            )

            self.blocks.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=channel_in,
                    out_channels=channel_out,
                    strides=2,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            )

        self.conv_out = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=num_channels[-1],
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def copy_weights_to_controlnet(controlnet: nn.Module, diffusion_model: nn.Module, verbose: bool = True) -> None:
    """
    Copy the state dict from the input diffusion model to the ControlNet, printing, if user requires it, the output
    keys that have matched and those that haven't.

    Args:
        controlnet: instance of ControlNet
        diffusion_model: instance of DiffusionModelUnet or SPADEDiffusionModelUnet
        verbose: if True, the matched and unmatched keys will be printed.
    """

    output = controlnet.load_state_dict(diffusion_model.state_dict(), strict=False)
    if verbose:
        dm_keys = [p[0] for p in list(diffusion_model.named_parameters()) if p[0] not in output.unexpected_keys]
        print(
            f"Copied weights from {len(dm_keys)} keys of the diffusion model into the ControlNet:"
            f"\n{'; '.join(dm_keys)}\nControlNet missing keys: {len(output.missing_keys)}:"
            f"\n{'; '.join(output.missing_keys)}\nDiffusion model incompatible keys: {len(output.unexpected_keys)}:"
            f"\n{'; '.join(output.unexpected_keys)}"
        )