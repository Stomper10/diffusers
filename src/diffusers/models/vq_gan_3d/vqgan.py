"""Adapted from https://github.com/SongweiGe/TATS"""
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import math
# import argparse
import numpy as np
# import pickle as pkl

# import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.distributed as dist

from .utils import shift_dim #, adopt_weight, comp_getattr
#from vq_gan_3d.model.lpips import LPIPS
from .codebook import Codebook

#from dataclasses import dataclass
from typing import Tuple #, Optional, Union

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
#from ...utils import BaseOutput
from ...utils.accelerate_utils import apply_forward_hook
#from ..autoencoders.vae import Decoder, DecoderOutput, Encoder, VectorQuantizer
from ..modeling_utils import ModelMixin


def silu(x):
    return x*torch.sigmoid(x)


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQGAN(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        embedding_dim: int = 256,
        n_codes: int = 2048,
        n_hiddens: int = 240,
        downsample: Tuple[int, ...] = (4, 4, 4),
        image_channels: int = 1,
        restart_thres: float = 1.0,
        no_random_restart=False,
        norm_type: str = "group",
        padding_type: str = "replicate",
        num_groups: int = 32,
    ):
        super().__init__()
        #self.cfg = cfg
        self.embedding_dim = embedding_dim
        self.n_codes = n_codes

        self.encoder = Encoder(n_hiddens, downsample, image_channels, norm_type, padding_type, num_groups,)
        self.decoder = Decoder(n_hiddens, downsample, image_channels, norm_type, num_groups)
        self.enc_out_ch = self.encoder.out_channels
        self.pre_vq_conv = SamePadConv3d(self.enc_out_ch, embedding_dim, 1, padding_type=padding_type)
        self.post_vq_conv = SamePadConv3d(embedding_dim, self.enc_out_ch, 1)

        self.codebook = Codebook(n_codes, embedding_dim, no_random_restart=no_random_restart, restart_thres=restart_thres)

        #self.gan_feat_weight = gan_feat_weight
        # TODO: Changed batchnorm from sync to normal
        #self.image_discriminator = NLayerDiscriminator(image_channels, disc_channels, disc_layers, norm_layer=nn.BatchNorm2d)
        #self.video_discriminator = NLayerDiscriminator3D(image_channels, disc_channels, disc_layers, norm_layer=nn.BatchNorm3d)

        #if disc_loss_type == 'vanilla':
        #    self.disc_loss = vanilla_d_loss
        #elif disc_loss_type == 'hinge':
        #    self.disc_loss = hinge_d_loss

        #self.perceptual_model = LPIPS().eval()

        #self.image_gan_weight = image_gan_weight
        #self.video_gan_weight = video_gan_weight
        #self.perceptual_weight = perceptual_weight
        #self.l1_weight = l1_weight
        #self.discriminator_iter_start = discriminator_iter_start
        #self.save_hyperparameters()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Encoder, Decoder)):
            module.gradient_checkpointing = value

    @apply_forward_hook
    def encode(self, x, include_embeddings=False, quantize=True):
        h = self.pre_vq_conv(self.encoder(x))
        if quantize:
            vq_output = self.codebook(h)
            if include_embeddings:
                return vq_output['embeddings'], vq_output['encodings']
            else:
                return vq_output['encodings']
        return h

    @apply_forward_hook
    def decode(self, latent, quantize=False):
        if quantize:
            vq_output = self.codebook(latent)
            latent = vq_output['encodings']
        h = F.embedding(latent, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h)

    def forward(self, x, optimizer_idx=None, log_image=False):
        B, C, T, H, W = x.shape

        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']))

        # print("### x.shape:", x.shape) # (B,1,56,40,40)
        # a = self.encoder(x)
        # print("### a.shape:", a.shape) # (B,32,28,20,20)
        # z = self.pre_vq_conv(a)
        # print("### z.shape:", z.shape) # (B,8,28,20,20)
        # vq_output = self.codebook(z)
        # w = vq_output['embeddings']
        # print("### w.shape:", w.shape) # (B,8,28,20,20)
        # s = self.post_vq_conv(w)
        # print("### s.shape:", s.shape) # (B,32,28,20,20)
        # x_recon = self.decoder(s)
        # print("### x_recon.shape:", x_recon.shape) # (B,1,56,40,40)

        return x_recon, vq_output

        # recon_loss = F.l1_loss(x_recon, x) * self.l1_weight

        # # Selects one random 2D image from each 3D Image
        # frame_idx = torch.randint(0, T, [B]).cuda()
        # frame_idx_selected = frame_idx.reshape(-1,
        #                                        1, 1, 1, 1).repeat(1, C, 1, H, W)
        # frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
        # frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)

        # if log_image:
        #     return frames, frames_recon, x, x_recon

        # if optimizer_idx == 0:
        #     # Autoencoder - train the "generator"

        #     # Perceptual loss
        #     perceptual_loss = 0
        #     if self.perceptual_weight > 0:
        #         perceptual_loss = self.perceptual_model(
        #             frames, frames_recon).mean() * self.perceptual_weight

        #     # Discriminator loss (turned on after a certain epoch)
        #     logits_image_fake, pred_image_fake = self.image_discriminator(
        #         frames_recon)
        #     logits_video_fake, pred_video_fake = self.video_discriminator(
        #         x_recon)
        #     g_image_loss = -torch.mean(logits_image_fake)
        #     g_video_loss = -torch.mean(logits_video_fake)
        #     g_loss = self.image_gan_weight*g_image_loss + self.video_gan_weight*g_video_loss
        #     disc_factor = adopt_weight(
        #         self.global_step, threshold=self.discriminator_iter_start)
        #     aeloss = disc_factor * g_loss

        #     # GAN feature matching loss - tune features such that we get the same prediction result on the discriminator
        #     image_gan_feat_loss = 0
        #     video_gan_feat_loss = 0
        #     feat_weights = 4.0 / (3 + 1)
        #     if self.image_gan_weight > 0:
        #         logits_image_real, pred_image_real = self.image_discriminator(
        #             frames)
        #         for i in range(len(pred_image_fake)-1):
        #             image_gan_feat_loss += feat_weights * \
        #                 F.l1_loss(pred_image_fake[i], pred_image_real[i].detach(
        #                 )) * (self.image_gan_weight > 0)
        #     if self.video_gan_weight > 0:
        #         logits_video_real, pred_video_real = self.video_discriminator(
        #             x)
        #         for i in range(len(pred_video_fake)-1):
        #             video_gan_feat_loss += feat_weights * \
        #                 F.l1_loss(pred_video_fake[i], pred_video_real[i].detach(
        #                 )) * (self.video_gan_weight > 0)
        #     gan_feat_loss = disc_factor * self.gan_feat_weight * \
        #         (image_gan_feat_loss + video_gan_feat_loss)

        #     self.log("train/g_image_loss", g_image_loss,
        #              logger=True, on_step=True, on_epoch=True)
        #     self.log("train/g_video_loss", g_video_loss,
        #              logger=True, on_step=True, on_epoch=True)
        #     self.log("train/image_gan_feat_loss", image_gan_feat_loss,
        #              logger=True, on_step=True, on_epoch=True)
        #     self.log("train/video_gan_feat_loss", video_gan_feat_loss,
        #              logger=True, on_step=True, on_epoch=True)
        #     self.log("train/perceptual_loss", perceptual_loss,
        #              prog_bar=True, logger=True, on_step=True, on_epoch=True)
        #     self.log("train/recon_loss", recon_loss, prog_bar=True,
        #              logger=True, on_step=True, on_epoch=True)
        #     self.log("train/aeloss", aeloss, prog_bar=True,
        #              logger=True, on_step=True, on_epoch=True)
        #     self.log("train/commitment_loss", vq_output['commitment_loss'],
        #              prog_bar=True, logger=True, on_step=True, on_epoch=True)
        #     self.log('train/perplexity', vq_output['perplexity'],
        #              prog_bar=True, logger=True, on_step=True, on_epoch=True)
        #     return recon_loss, x_recon, vq_output, aeloss, perceptual_loss, gan_feat_loss

        # if optimizer_idx == 1:
        #     # Train discriminator
        #     logits_image_real, _ = self.image_discriminator(frames.detach())
        #     logits_video_real, _ = self.video_discriminator(x.detach())

        #     logits_image_fake, _ = self.image_discriminator(
        #         frames_recon.detach())
        #     logits_video_fake, _ = self.video_discriminator(x_recon.detach())

        #     d_image_loss = self.disc_loss(logits_image_real, logits_image_fake)
        #     d_video_loss = self.disc_loss(logits_video_real, logits_video_fake)
        #     disc_factor = adopt_weight(
        #         self.global_step, threshold=self.discriminator_iter_start)
        #     discloss = disc_factor * \
        #         (self.image_gan_weight*d_image_loss +
        #          self.video_gan_weight*d_video_loss)

        #     self.log("train/logits_image_real", logits_image_real.mean().detach(),
        #              logger=True, on_step=True, on_epoch=True)
        #     self.log("train/logits_image_fake", logits_image_fake.mean().detach(),
        #              logger=True, on_step=True, on_epoch=True)
        #     self.log("train/logits_video_real", logits_video_real.mean().detach(),
        #              logger=True, on_step=True, on_epoch=True)
        #     self.log("train/logits_video_fake", logits_video_fake.mean().detach(),
        #              logger=True, on_step=True, on_epoch=True)
        #     self.log("train/d_image_loss", d_image_loss,
        #              logger=True, on_step=True, on_epoch=True)
        #     self.log("train/d_video_loss", d_video_loss,
        #              logger=True, on_step=True, on_epoch=True)
        #     self.log("train/discloss", discloss, prog_bar=True,
        #              logger=True, on_step=True, on_epoch=True)
        #     return discloss

        # perceptual_loss = self.perceptual_model(
        #     frames, frames_recon) * self.perceptual_weight
        # return recon_loss, x_recon, vq_output, perceptual_loss

    # def training_step(self, batch, batch_idx, optimizer_idx):
    #     x = batch['data']
    #     if optimizer_idx == 0:
    #         recon_loss, _, vq_output, aeloss, perceptual_loss, gan_feat_loss = self.forward(
    #             x, optimizer_idx)
    #         commitment_loss = vq_output['commitment_loss']
    #         loss = recon_loss + commitment_loss + aeloss + perceptual_loss + gan_feat_loss
    #     if optimizer_idx == 1:
    #         discloss = self.forward(x, optimizer_idx)
    #         loss = discloss
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     x = batch['data']  # TODO: batch['stft']
    #     recon_loss, _, vq_output, perceptual_loss = self.forward(x)
    #     self.log('val/recon_loss', recon_loss, prog_bar=True)
    #     self.log('val/perceptual_loss', perceptual_loss, prog_bar=True)
    #     self.log('val/perplexity', vq_output['perplexity'], prog_bar=True)
    #     self.log('val/commitment_loss',
    #              vq_output['commitment_loss'], prog_bar=True)

    # def configure_optimizers(self):
    #     lr = self.cfg.model.lr
    #     opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
    #                               list(self.decoder.parameters()) +
    #                               list(self.pre_vq_conv.parameters()) +
    #                               list(self.post_vq_conv.parameters()) +
    #                               list(self.codebook.parameters()),
    #                               lr=lr, betas=(0.5, 0.9))
    #     opt_disc = torch.optim.Adam(list(self.image_discriminator.parameters()) +
    #                                 list(self.video_discriminator.parameters()),
    #                                 lr=lr, betas=(0.5, 0.9))
    #     return [opt_ae, opt_disc], []

    # def log_images(self, batch, **kwargs):
    #     log = dict()
    #     x = batch['data']
    #     x = x.to(self.device)
    #     frames, frames_rec, _, _ = self(x, log_image=True)
    #     log["inputs"] = frames
    #     log["reconstructions"] = frames_rec
    #     #log['mean_org'] = batch['mean_org']
    #     #log['std_org'] = batch['std_org']
    #     return log

    # def log_videos(self, batch, **kwargs):
    #     log = dict()
    #     x = batch['data']
    #     _, _, x, x_rec = self(x, log_image=True)
    #     log["inputs"] = x
    #     log["reconstructions"] = x_rec
    #     #log['mean_org'] = batch['mean_org']
    #     #log['std_org'] = batch['std_org']
    #     return log


def Normalize(in_channels, norm_type='group', num_groups=32):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        # TODO Changed num_groups from 32 to 8
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)


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


class Decoder(nn.Module):
    def __init__(self, n_hiddens, upsample, image_channel, norm_type='group', num_groups=32):
        super().__init__()

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()

        in_channels = n_hiddens*2**max_us
        self.final_block = nn.Sequential(
            Normalize(in_channels, norm_type, num_groups=num_groups),
            SiLU()
        )

        self.conv_blocks = nn.ModuleList()
        for i in range(max_us):
            block = nn.Module()
            in_channels = in_channels if i == 0 else n_hiddens*2**(max_us-i+1)
            out_channels = n_hiddens*2**(max_us-i)
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            block.up = SamePadConvTranspose3d(
                in_channels, out_channels, 4, stride=us)
            block.res1 = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            block.res2 = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            self.conv_blocks.append(block)
            n_times_upsample -= 1

        self.conv_last = SamePadConv3d(
            out_channels, image_channel, kernel_size=3)

    def forward(self, x):
        h = self.final_block(x)
        for i, block in enumerate(self.conv_blocks):
            h = block.up(h)
            h = block.res1(h)
            h = block.res2(h)
        h = self.conv_last(h)
        return h


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


# Does not support dilation
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


class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input, mode=self.padding_type))


class NLayerDiscriminator(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self, 
        image_channels: int = 3, 
        disc_channels: int = 64, 
        disc_layers: int = 3, 
        norm_layer: str = "BatchNorm2d", 
        use_sigmoid=False, 
        getIntermFeat=True
    ):
        # def __init__(self, image_channels, disc_channels=64, disc_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.disc_layers = disc_layers
        if norm_layer == "BatchNorm2d":
            norm_layer = nn.BatchNorm2d
        else:
            raise ValueError(f"unexpected norm layer type: {norm_layer}")

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(image_channels, disc_channels, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = disc_channels
        for n in range(1, disc_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.disc_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _


class NLayerDiscriminator3D(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self, 
        image_channels: int = 3, 
        disc_channels: int = 64, 
        disc_layers: int = 3, 
        norm_layer: str = "BatchNorm3d", 
        use_sigmoid=False, 
        getIntermFeat=True
    ):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.disc_layers = disc_layers
        if norm_layer == "BatchNorm3d":
            norm_layer = nn.BatchNorm3d
        else:
            raise ValueError(f"unexpected norm layer type: {norm_layer}")

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv3d(image_channels, disc_channels, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = disc_channels
        for n in range(1, disc_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.disc_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _

#########################

class Mapper3D(nn.Module):
    """
    A 3D Mapper network that:
      1) Maps (1,28,20,20) -> (8,28,20,20) via an initial conv,
      2) Passes through multiple residual blocks at 8 channels,
      3) Outputs a final latent feature map (8,28,20,20).

    Args:
        in_channels (int): Input channels. Default=1 for MRI.
        out_channels (int): Output channels (latent dim). Default=8.
        num_resblocks (int): How many residual blocks to stack.
                             Each block keeps the same shape.
    """
    def __init__(
        self,
        in_channels=1,
        n_hiddens=4,
        out_channels=8,
        norm_type='group',
        padding_type='replicate',
        num_groups=8,
    ):
        super().__init__()

        self.conv1 = SamePadConv3d(
            in_channels, n_hiddens, kernel_size=3, padding_type=padding_type)
        self.conv2 = SamePadConv3d(
            n_hiddens, out_channels, kernel_size=3, padding_type=padding_type)
        self.res = ResBlock(
            out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type, num_groups=num_groups),
            SiLU()
        )

    def forward(self, x):
        """
        x: Tensor (B,1,28,20,20)
        returns: Tensor (B,8,28,20,20)
        """
        x = self.conv1(x)        # => (B,4,28,20,20)
        x = self.conv2(x)        # => (B,4,28,20,20)
        x = self.res(x)          # => (B,8,28,20,20)
        x = self.final_block(x)  # => (B,8,28,20,20)
        return x
    
# class Map_Block(nn.Module):
#     def __init__(self, in_channels, n_hiddens, out_channels, norm_type='group', padding_type='replicate', num_groups=8):
#         super().__init__()

#         self.conv1 = SamePadConv3d(
#             in_channels, n_hiddens, kernel_size=3, padding_type=padding_type)
#         self.conv2 = SamePadConv3d(
#             n_hiddens, out_channels, kernel_size=3, padding_type=padding_type)
#         self.res = ResBlock(
#             out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
#         self.final_block = nn.Sequential(
#             Normalize(out_channels, norm_type, num_groups=num_groups),
#             SiLU()
#         )

#     def forward(self, x):
#         """
#         x: Tensor (B,1,28,20,20)
#         returns: Tensor (B,8,28,20,20)
#         """
#         x = self.conv1(x)        # => (B,4,28,20,20)
#         x = self.conv2(x)        # => (B,4,28,20,20)
#         x = self.res(x)          # => (B,8,28,20,20)
#         x = self.final_block(x)  # => (B,8,28,20,20)
#         return x
    

# class SRVQGAN(ModelMixin, ConfigMixin):
#     """
#     Super-resolution by leveraging a pretrained 3D VQGAN decoder.
#     LR -> Mapper -> post_vq_conv -> Decoder -> HR
#     """
#     def __init__(
#         self,
#         vqgan,              # the pretrained VQGAN instance
#         mapper,             # Mapper3D instance
#         freeze_decoder=True # if True, we freeze VQGAN decoder & post_vq_conv
#     ):
#         super().__init__()
#         self.vqgan = vqgan
#         self.mapper = mapper

#         if freeze_decoder:
#             self.vqgan.decoder.requires_grad_(False)
#             self.vqgan.post_vq_conv.requires_grad_(False)
#             # self.vqgan.codebook.requires_grad_(False)  # typically not used if we feed embeddings directly
#             # self.vqgan.encoder.requires_grad_(False)    # not used in the forward pass anyway

#     def forward(self, x_lr):
#         """
#         x_lr: (B,1,D_lr,H_lr,W_lr) Low-res patch
#         Returns: x_sr: (B,1,D_hr,H_hr,W_hr) Up-sampled patch
#         """
#         # 1) Map LR patch to latent embeddings
#         latents = self.mapper(x_lr)  
#         # shape: (B, embedding_dim, D_latent, H_latent, W_latent)

#         # 2) Pass latents through post_vq_conv
#         #    i.e. "reverse" of pre_vq_conv
#         latents_post = self.vqgan.post_vq_conv(latents)
#         # shape: (B, enc_out_ch, D_latent, H_latent, W_latent)

#         # 3) Decode to high-res
#         x_sr = self.vqgan.decoder(latents_post)  # (B,1,D_hr,H_hr,W_hr)
#         return x_sr
    

class SRVQGAN(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        embedding_dim: int = 256,
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
        #self.cfg = cfg
        self.embedding_dim = embedding_dim
        # self.n_codes = n_codes

        self.encoder = Encoder(n_hiddens, downsample, image_channels, norm_type, padding_type, num_groups,)
        self.encoder.requires_grad_(False)
        self.decoder = Decoder(n_hiddens, downsample, image_channels, norm_type, num_groups)
        self.enc_out_ch = self.encoder.out_channels
        # self.pre_vq_conv = SamePadConv3d(self.enc_out_ch, embedding_dim, 1, padding_type=padding_type)
        self.post_vq_conv = SamePadConv3d(embedding_dim, self.enc_out_ch, 1)

        # self.codebook = Codebook(n_codes, embedding_dim, no_random_restart=no_random_restart, restart_thres=restart_thres)

        self.mapper = Mapper3D(in_channels=image_channels, 
                               n_hiddens=embedding_dim//2, 
                               out_channels=embedding_dim, 
                               norm_type=norm_type, 
                               padding_type=padding_type, 
                               num_groups=embedding_dim,)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Mapper3D, Decoder)):
            module.gradient_checkpointing = value

    def forward(self, x_lr):
        """
        x_lr: (B,1,D_lr,H_lr,W_lr) Low-res patch
        Returns: x_sr: (B,1,D_hr,H_hr,W_hr) Up-sampled patch
        """
        # 1) Map LR patch to latent embeddings
        latents = self.mapper(x_lr)  
        # shape: (B, embedding_dim, D_latent, H_latent, W_latent)

        # 2) Pass latents through post_vq_conv
        #    i.e. "reverse" of pre_vq_conv
        latents_post = self.post_vq_conv(latents)
        # shape: (B, enc_out_ch, D_latent, H_latent, W_latent)

        # 3) Decode to high-res
        x_sr = self.decoder(latents_post)  # (B,1,D_hr,H_hr,W_hr)
        return x_sr

########################
# File: src/diffusers/models/super_resolution_2plus1d.py

class Conv2p1D(nn.Module):
    """
    (2+1)D convolution:
    First a 2D conv over spatial dims, then a 1D conv over the 'temporal' dim.
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)):
        super().__init__()
        # Split kernel_size into (1, ky, kx) and (kt, 1, 1)
        # Example: (3,3,3) -> first is (1,3,3) then (3,1,1)
        mid_channels = out_channels
        self.conv2d = nn.Conv3d(
            in_channels,
            mid_channels,
            kernel_size=(1, kernel_size[1], kernel_size[2]),
            padding=(0, padding[1], padding[2])
        )
        self.conv1d = nn.Conv3d(
            mid_channels,
            out_channels,
            kernel_size=(kernel_size[0], 1, 1),
            padding=(padding[0], 0, 0)
        )

    def forward(self, x):
        x = self.conv2d(x)
        x = F.relu(x)
        x = self.conv1d(x)
        return x

class DownBlock(nn.Module):
    """
    Example Down-sampling block using (2+1)D convs and a stride of 2 in spatial dims.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2p1D(in_channels, out_channels)
        self.conv2 = Conv2p1D(out_channels, out_channels)
        # For downsampling, we apply stride in the spatial dims (H,W)
        self.down = nn.Conv3d(out_channels, out_channels, kernel_size=(1,2,2), stride=(1,2,2))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x_down = self.down(x)  # downsample spatially
        return x, x_down

class UpBlock(nn.Module):
    """
    Example Up-sampling block using (2+1)D convs and upsampling in spatial dims.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2p1D(in_channels, out_channels)
        self.conv2 = Conv2p1D(out_channels, out_channels)

    def forward(self, x, skip):
        # Upsample in the spatial dims (H,W) by factor of 2
        x = F.interpolate(x, scale_factor=(1,2,2), mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)  # Skip connection
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x

class SuperResolutionUNet3D(ModelMixin, ConfigMixin):
    
    _supports_gradient_checkpointing = True
    
    """
    A simplified (2+1)D U-Net for super-resolution from 224x40x40 -> 224x160x160.
    """
    @register_to_config
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()

        # Down blocks
        self.down1 = DownBlock(in_channels, base_channels)        # 1 -> 32
        self.down2 = DownBlock(base_channels, base_channels * 2)  # 32 -> 64
        self.down3 = DownBlock(base_channels * 2, base_channels * 4)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            Conv2p1D(base_channels * 4, base_channels * 8),
            nn.ReLU(),
            Conv2p1D(base_channels * 8, base_channels * 4),
            nn.ReLU()
        )

        # Up blocks
        self.up3 = UpBlock(base_channels * 8, base_channels * 2)
        self.up2 = UpBlock(base_channels * 4, base_channels)
        self.up1 = UpBlock(base_channels * 2, base_channels)

        # Final conv to get desired output
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def _set_gradient_checkpointing(self, module, value=False):
        module.gradient_checkpointing = value

    def forward(self, x):
        """
        x shape: (B, in_channels=1, T=224, H=40, W=40)
        output shape: (B, out_channels=1, T=224, H=160, W=160)
        """
        # Down-sampling
        skip1, d1 = self.down1(x)     # skip1: (B,32,224,40,40), d1: (B,32,224,20,20)
        skip2, d2 = self.down2(d1)    # skip2: (B,64,224,20,20), d2: (B,64,224,10,10)
        skip3, d3 = self.down3(d2)    # skip3: (B,128,224,10,10), d3: (B,128,224,5,5)

        # Bottleneck
        b = self.bottleneck(d3)       # (B,128,224,5,5)

        # Up-sampling
        up3 = self.up3(b, skip3)      # -> (B,64,224,10,10)
        up2 = self.up2(up3, skip2)    # -> (B,32,224,20,20)
        up1 = self.up1(up2, skip1)    # -> (B,32,224,40,40)

        # Final upsampling to 4x in H,W (because 40->160)
        # We already did 2×2×2 up and 2×2×2 up in the blocks, 
        # so each block doubles spatial dims. If you need 
        # exactly x4 from the final block, you can do:
        out = F.interpolate(up1, scale_factor=(1,4,4), mode='trilinear', align_corners=False)
        
        # Output conv
        out = self.out_conv(out)
        return out
    
class SuperResolutionUNet3D_Trans(ModelMixin, ConfigMixin):
    
    _supports_gradient_checkpointing = True
    
    """
    A simplified (2+1)D U-Net for super-resolution from 224x40x40 -> 224x160x160.
    """
    @register_to_config
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()

        # Down blocks
        self.down1 = DownBlock(in_channels, base_channels)        # 1 -> 32
        self.down2 = DownBlock(base_channels, base_channels * 2)  # 32 -> 64
        self.down3 = DownBlock(base_channels * 2, base_channels * 4)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            Conv2p1D(base_channels * 4, base_channels * 8),
            nn.ReLU(),
            Conv2p1D(base_channels * 8, base_channels * 4),
            nn.ReLU()
        )

        # Up blocks
        self.up3 = UpBlock(base_channels * 8, base_channels * 2)
        self.up2 = UpBlock(base_channels * 4, base_channels)
        self.up1 = UpBlock(base_channels * 2, base_channels)

        self.upsample_tconv1 = nn.ConvTranspose3d(
            in_channels=base_channels,  # or whatever your up_1 channels are
            out_channels=base_channels, 
            kernel_size=(1,4,4), 
            stride=(1,2,2), 
            padding=(0,1,1)
        )
        self.upsample_tconv2 = nn.ConvTranspose3d(
            in_channels=base_channels,
            out_channels=out_channels,
            kernel_size=(1,4,4),
            stride=(1,2,2),
            padding=(0,1,1)
        )

        # Final conv to get desired output
        #self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def _set_gradient_checkpointing(self, module, value=False):
        module.gradient_checkpointing = value

    def forward(self, x):
        """
        x shape: (B, in_channels=1, T=224, H=40, W=40)
        output shape: (B, out_channels=1, T=224, H=160, W=160)
        """
        # Down-sampling
        skip1, d1 = self.down1(x)     # skip1: (B,32,224,40,40), d1: (B,32,224,20,20)
        skip2, d2 = self.down2(d1)    # skip2: (B,64,224,20,20), d2: (B,64,224,10,10)
        skip3, d3 = self.down3(d2)    # skip3: (B,128,224,10,10), d3: (B,128,224,5,5)

        # Bottleneck
        b = self.bottleneck(d3)       # (B,128,224,5,5)

        # Up-sampling
        up3 = self.up3(b, skip3)      # -> (B,64,224,10,10)
        up2 = self.up2(up3, skip2)    # -> (B,32,224,20,20)
        up1 = self.up1(up2, skip1)    # -> (B,32,224,40,40)

        # Final upsampling to 4x in H,W (because 40->160)
        # We already did 2×2×2 up and 2×2×2 up in the blocks, 
        # so each block doubles spatial dims. If you need 
        # exactly x4 from the final block, you can do:
        #out = F.interpolate(up1, scale_factor=(1,4,4), mode='trilinear', align_corners=False)
        x = self.upsample_tconv1(up1)
        x = F.relu(x, inplace=True)

        # Second transposed conv: 80 -> 160
        out = self.upsample_tconv2(x)
        
        # Output conv
        #out = self.out_conv(out)
        return out

########################

# ------------------------------------------------------------------------------
# 1) (2+1)D Convolution + a simple Self-Attention for 3D volumes
# ------------------------------------------------------------------------------

class SimpleSelfAttention3D(nn.Module):
    """
    A straightforward 3D self-attention mechanism.
    We'll flatten (T*H*W) as "sequence" and do standard QKV attention,
    but keep the channel dimension separate.

    For large T, H, W, this can be memory-heavy. We keep it simpler by:
      - Possibly reducing channels with a bottleneck
      - or using a small head dimension
    """
    def __init__(self, channels, head_dim=64):
        super().__init__()
        self.channels = channels
        self.head_dim = head_dim
        self.num_heads = channels // head_dim

        # Project to queries, keys, values
        self.q_proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv3d(channels, channels, kernel_size=1)

        # Final linear layer
        self.out_proj = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape

        # Flatten (T,H,W) -> sequence dimension S = T*H*W
        S = T * H * W
        
        # 1) Compute Q, K, V
        q = self.q_proj(x)  # (B, C, T, H, W)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2) Reshape for multi-head attention
        #    => (B, num_heads, head_dim, S)
        def reshape_to_heads(tensor):
            return tensor.view(B, self.num_heads, self.head_dim, S)

        q = reshape_to_heads(q)
        k = reshape_to_heads(k)
        v = reshape_to_heads(v)

        # 3) Transpose for easier dot product:
        #    Q => (B, num_heads, S, head_dim)
        #    K => (B, num_heads, head_dim, S)
        q = q.permute(0, 1, 3, 2)  # (B, heads, S, head_dim)
        k = k.permute(0, 1, 2, 3)  # (B, heads, head_dim, S)

        # 4) Compute attention scores
        attn_scores = torch.matmul(q, k) / (self.head_dim ** 0.5)
        # => shape (B, heads, S, S)

        attn_probs = F.softmax(attn_scores, dim=-1)  # (B, heads, S, S)

        # 5) Apply to V
        #    V => (B, heads, head_dim, S), so permute similarly:
        v = v.permute(0, 1, 3, 2)  # (B, heads, S, head_dim)
        out = torch.matmul(attn_probs, v)  # (B, heads, S, head_dim)

        # 6) Reshape back to (B,C,T,H,W)
        out = out.permute(0,1,3,2).contiguous()  # (B, heads, head_dim, S)
        out = out.view(B, C, T, H, W)

        out = self.out_proj(out)  # (B, C, T, H, W)
        return out

# ------------------------------------------------------------------------------
# 2) DownBlock / UpBlock with optional Self-Attention
# ------------------------------------------------------------------------------

class DownBlockAttn(nn.Module):
    """
    Each DownBlock does:
      - Two (2+1)D convolutions
      - Optional attention
      - Downsample in spatial dims by factor of 2
    """
    def __init__(self, in_channels, out_channels, use_attn=False):
        super().__init__()
        self.conv1 = Conv2p1D(in_channels, out_channels)
        self.conv2 = Conv2p1D(out_channels, out_channels)
        self.use_attn = use_attn
        self.attn = SimpleSelfAttention3D(out_channels) if use_attn else None
        # Downsample (stride=2 on H,W), keep T the same
        self.down = nn.Conv3d(out_channels, out_channels,
                              kernel_size=(1,2,2), stride=(1,2,2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_attn:
            x = self.attn(x)
        skip = x
        down = self.down(x)
        return skip, down


class UpBlockAttn(nn.Module):
    """
    Each UpBlock does:
      - Upsample in spatial dims by factor of 2
      - Concatenate skip
      - Two (2+1)D convolutions
      - Optional attention
    """
    def __init__(self, in_channels, out_channels, use_attn=False):
        super().__init__()
        self.use_attn = use_attn
        self.conv1 = Conv2p1D(in_channels, out_channels)
        self.conv2 = Conv2p1D(out_channels, out_channels)
        self.attn = SimpleSelfAttention3D(out_channels) if use_attn else None

    def forward(self, x, skip):
        # Upsample (spatial dims only)
        x = F.interpolate(x, scale_factor=(1,2,2), mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_attn:
            x = self.attn(x)
        return x

# ------------------------------------------------------------------------------
# 3) The ComplexSuperResolutionUNet3D model
# ------------------------------------------------------------------------------

class ComplexSuperResolutionUNet3D(ModelMixin, ConfigMixin):
    """
    A deeper (2+1)D U-Net with attention blocks to handle more capacity.
    - Input shape: (B, in_channels, T=224, H=40, W=40)
    - Output shape: (B, out_channels, T=224, H=160, W=160)
    """
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 base_channels=64):
        super().__init__()

        # Down: 4 levels
        # Optionally add attention at certain stages, e.g. after the second or third level
        self.down1 = DownBlockAttn(in_channels, base_channels, use_attn=False)           # 40->20
        self.down2 = DownBlockAttn(base_channels, base_channels*2, use_attn=False)       # 20->10
        self.down3 = DownBlockAttn(base_channels*2, base_channels*4, use_attn=True)     # 10->5
        #self.down4 = DownBlockAttn(base_channels*4, base_channels*8, use_attn=True)    # 5->2 or 3? (Note: 5->2 if stride=2, int division)

        # Bottleneck: apply heavy convs + attention
        self.bottleneck = nn.Sequential(
            Conv2p1D(base_channels*4, base_channels*4),
            SimpleSelfAttention3D(base_channels*4),
            Conv2p1D(base_channels*4, base_channels*4),
        )

        # Up: 4 levels
        # Combine skip connections
        #self.up4 = UpBlockAttn(base_channels*8 + base_channels*8, base_channels*4, use_attn=True)
        self.up3 = UpBlockAttn(base_channels*4 + base_channels*4, base_channels*2, use_attn=True)
        self.up2 = UpBlockAttn(base_channels*2 + base_channels*2, base_channels, use_attn=False)
        self.up1 = UpBlockAttn(base_channels + base_channels, base_channels, use_attn=False)

        # Final 4× upsampling
        self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward_3d(self, x):
        """
        The main forward pass for the entire volume
        x shape: (B,1,T,H=40,W=40)
        We want: (B,1,T,H=160,W=160)
        """

        skip1, d1 = self.down1(x)   # skip1->(B,64,T,40,40), d1->(B,64,T,20,20)
        skip2, d2 = self.down2(d1)  # skip2->(B,128,T,20,20), d2->(B,128,T,10,10)
        skip3, d3 = self.down3(d2)  # skip3->(B,256,T,10,10), d3->(B,256,T,5,5)
        #skip4, d4 = self.down4(d3)  # skip4->(B,512,T,5,5),   d4->(B,512,T,2,2) or (2,3)

        b = self.bottleneck(d3)     # (B,512,T,2,2) or (2,3) depending on shape

        # Up path
        #u4 = self.up4(b, skip4)     # -> (B,256,T,5,5)
        u3 = self.up3(b, skip3)    # -> (B,128,T,10,10)
        u2 = self.up2(u3, skip2)    # -> (B,64,T,20,20)
        u1 = self.up1(u2, skip1)    # -> (B,64,T,40,40)

        # We have 40×40, but need 160×160 => 4× up
        out = F.interpolate(u1, scale_factor=(1,4,4), mode='trilinear', align_corners=False)
        out = self.final_conv(out)  # => (B,1,T,160,160)
        return out

    def forward_in_chunks(self, x, chunk_size=32, overlap_size=8):
        """
        Process x along the T dimension in overlapping chunks, blending them
        in a fully differentiable way so gradients can flow back through each chunk.

        x shape: (B, C, T, H, W)
        returns shape: (B, out_channels, T, 4H, 4W)
        """
        device = x.device
        B, C, T, H, W = x.shape

        # 1) Figure out output shape by running a "dummy" chunk.
        #    Remove the no_grad() so that PyTorch won't break the graph later,
        #    but typically this dummy pass won't consume much memory if chunk_size is small.
        dummy_chunk_len = min(chunk_size, T)  # in case T < chunk_size
        dummy_out = self.forward_3d(x[:, :, :dummy_chunk_len, :, :])
        out_channels = dummy_out.shape[1]
        up_H = dummy_out.shape[3]  # if you're doing 4× upsampling: up_H = 4*H
        up_W = dummy_out.shape[4]  # up_W = 4*W

        # 2) We'll accumulate per-frame outputs in Python lists, one entry for each of T frames.
        #    Each entry is a tensor of shape (B, out_channels, up_H, up_W).
        #    We'll also keep a "weight" tensor for each frame to handle overlapping blends.
        out_frames = [None] * T
        weight_frames = [None] * T

        # Initialize each frame's accumulator
        for t in range(T):
            out_frames[t] = torch.zeros(
                (B, out_channels, up_H, up_W),
                dtype=x.dtype, device=device
            )
            # We'll store scalar weights per frame, but we need it to broadcast over (B, outC, upH, upW)
            weight_frames[t] = torch.zeros(
                (B, 1, 1, 1),
                dtype=x.dtype, device=device
            )

        # 3) Compute chunk boundaries
        chunk_starts = list(range(0, T, chunk_size))

        for i, start_t in enumerate(chunk_starts):
            end_t = start_t + chunk_size + overlap_size
            if end_t > T:
                end_t = T

            # Extract chunk
            x_chunk = x[:, :, start_t:end_t, :, :]  # shape: (B,C,chunk_len,H,W)
            chunk_len = x_chunk.shape[2]

            # Forward pass for this chunk
            out_chunk = self.forward_3d(x_chunk)  # (B, out_channels, chunk_len, up_H, up_W)

            # 4) Build a per-frame weight vector for blending
            chunk_weight = torch.ones(chunk_len, device=device, dtype=x.dtype)

            # Overlap region at the start if i>0
            overlap = min(overlap_size, chunk_len)
            if i > 0 and overlap > 0:
                # Fade in: linearly ramp from 0 -> 1 across the first `overlap` frames
                for t_idx in range(overlap):
                    alpha = float(t_idx + 1) / overlap  # e.g. 1/overlap, 2/overlap, ...
                    chunk_weight[t_idx] = alpha

            # Overlap region at the end if there's another chunk after this
            if (start_t + chunk_size) < T:
                for t_idx in range(chunk_len - overlap, chunk_len):
                    idx_from_end = (chunk_len - 1) - t_idx
                    alpha = float(idx_from_end) / overlap
                    chunk_weight[t_idx] = alpha

            # 5) Accumulate the results frame-by-frame
            #    out_chunk is shape (B, outC, chunk_len, up_H, up_W)
            #    chunk_weight is shape (chunk_len,)
            # We'll multiply out_chunk[:, :, frame_offset] by chunk_weight[frame_offset]
            # and add it to out_frames[t_global].
            for frame_offset in range(chunk_len):
                t_global = start_t + frame_offset
                w = chunk_weight[frame_offset]  # scalar

                # Add to the frame accumulator
                out_frames[t_global] += out_chunk[:, :, frame_offset] * w
                # Accumulate weight
                weight_frames[t_global] += w

        # 6) Normalize each frame by its total weight
        final_frames = []
        for t in range(T):
            w = weight_frames[t]
            # Avoid division by zero
            w_clamped = torch.clamp(w, min=1e-8)
            # out_frames[t] shape: (B, out_channels, up_H, up_W)
            # w shape: (B, 1, 1, 1) => broadcasts
            normalized_frame = out_frames[t] / w_clamped
            final_frames.append(normalized_frame.unsqueeze(2))
            # unsqueeze(2) to insert time dimension => (B, outC, 1, up_H, up_W)

        # 7) Stack along the time dimension
        #    final shape => (B, outC, T, up_H, up_W)
        out_video = torch.cat(final_frames, dim=2)

        return out_video
    
    def _set_gradient_checkpointing(self, module, value=False):
        module.gradient_checkpointing = value

    def forward(self, x, chunk_size=32, overlap_size=8):
        if x.shape[2] <= chunk_size + overlap_size:
            return self.forward_3d(x)
        else:
            return self.forward_in_chunks(x, chunk_size=chunk_size, overlap_size=overlap_size)


class SRUNET3D(ModelMixin, ConfigMixin):
    """
    A deeper (2+1)D U-Net with attention blocks to handle more capacity.
    - Input shape: (B, in_channels, T=224, H=40, W=40)
    - Output shape: (B, out_channels, T=224, H=160, W=160)
    """
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 base_channels=64):
        super().__init__()

        # Down: 3 or 4 levels
        # Optionally add attention at certain stages, e.g. after the second or third level
        self.down1 = DownBlockAttn(in_channels, base_channels, use_attn=False)       # 40->20 / 64->32
        self.down2 = DownBlockAttn(base_channels, base_channels*2, use_attn=False)   # 20->10 / 32->16
        self.down3 = DownBlockAttn(base_channels*2, base_channels*4, use_attn=True)  # 10->5  / 16->8
        self.down4 = DownBlockAttn(base_channels*4, base_channels*8, use_attn=True)     # 8->4
        #self.down5 = DownBlockAttn(base_channels*8, base_channels*16, use_attn=True) # SET

        # Bottleneck: apply heavy convs + attention
        self.bottleneck = nn.Sequential(
            Conv2p1D(base_channels*8, base_channels*8), # *8
            SimpleSelfAttention3D(base_channels*8),
            Conv2p1D(base_channels*8, base_channels*8),
        )

        # Up: 3 or 4 levels
        # Combine skip connections
        #self.up5 = UpBlockAttn(base_channels*16 + base_channels*16, base_channels*8, use_attn=True)       # 4->8
        self.up4 = UpBlockAttn(base_channels*8 + base_channels*8, base_channels*4, use_attn=True)       # 4->8
        self.up3 = UpBlockAttn(base_channels*4 + base_channels*4, base_channels*2, use_attn=True)   # 5->10 / 8->16
        self.up2 = UpBlockAttn(base_channels*2 + base_channels*2, base_channels, use_attn=False)    # 10->20 / 16->32
        self.up1 = UpBlockAttn(base_channels + base_channels, base_channels, use_attn=False)        # 20->40 / 32->64

        self.upsample_tconv1 = nn.ConvTranspose3d(
            in_channels=base_channels,  # or whatever your up_1 channels are
            out_channels=base_channels, 
            kernel_size=(1,4,4), 
            stride=(1,2,2), 
            padding=(0,1,1)
        )
        self.upsample_tconv2 = nn.ConvTranspose3d(
            in_channels=base_channels,
            out_channels=out_channels,
            kernel_size=(1,4,4),
            stride=(1,2,2),
            padding=(0,1,1)
        )

    def forward_3d(self, x):
        """
        The main forward pass for the entire volume
        x shape: (B,1,T,H=40,W=40)
        We want: (B,1,T,H=160,W=160)
        """

        skip1, d1 = self.down1(x)   # skip1 (B, 64,T,48,48), d1 (B, 64,T,24,24) / skip1 (B, 64,T,64,64), d1 (B, 64,T,32,32)
        skip2, d2 = self.down2(d1)  # skip2 (B,128,T,24,24), d2 (B,128,T,12,12) / skip2 (B,128,T,32,32), d2 (B,128,T,16,16)
        skip3, d3 = self.down3(d2)  # skip3 (B,256,T,12,12), d3 (B,256,T, 6, 6) / skip3 (B,256,T,16,16), d3 (B,256,T, 8, 8)
        skip4, d4 = self.down4(d3)  # skip4 (B,512,T, 6, 6), d4 (B,512,T, 3, 3) / skip3 (B,256,T, 8, 8), d3 (B,256,T, 4, 4)
                        
        b = self.bottleneck(d4)     # (B,512,T,4,4)
        
        # Up path
        u4 = self.up4(b, skip4)     # (B,256,T,8,8)
        u3 = self.up3(u4, skip3)    # (B,128,T,10,10) / (B,128,T,16,16)
        u2 = self.up2(u3, skip2)    # (B, 64,T,20,20) / (B, 64,T,32,32)
        u1 = self.up1(u2, skip1)    # (B, 64,T,40,40) / (B, 64,T,64,64)
        
        if x.shape[-1] == 64:
            out = self.upsample_tconv2(u1) # (B,1,T,160,160)
            return out

        # need 4x up
        x = self.upsample_tconv1(u1) # (B,64,T,80,80)
        x = F.relu(x, inplace=True)
        out = self.upsample_tconv2(x) # (B,1,T,160,160)
        return out

    def forward_in_chunks(self, x, chunk_size=32, overlap_size=8):
        """
        Process x along the T dimension in overlapping chunks, blending them
        in a fully differentiable way so gradients can flow back through each chunk.

        x shape: (B, C, T, H, W)
        returns shape: (B, out_channels, T, 4H, 4W)
        """
        device = x.device
        B, C, T, H, W = x.shape

        # 1) Figure out output shape by running a "dummy" chunk.
        #    Remove the no_grad() so that PyTorch won't break the graph later,
        #    but typically this dummy pass won't consume much memory if chunk_size is small.
        dummy_chunk_len = min(chunk_size, T)  # in case T < chunk_size
        dummy_out = self.forward_3d(x[:, :, :dummy_chunk_len, :, :])
        out_channels = dummy_out.shape[1]
        up_H = dummy_out.shape[3]  # if you're doing 4× upsampling: up_H = 4*H
        up_W = dummy_out.shape[4]  # up_W = 4*W

        # 2) We'll accumulate per-frame outputs in Python lists, one entry for each of T frames.
        #    Each entry is a tensor of shape (B, out_channels, up_H, up_W).
        #    We'll also keep a "weight" tensor for each frame to handle overlapping blends.
        out_frames = [None] * T
        weight_frames = [None] * T

        # Initialize each frame's accumulator
        for t in range(T):
            out_frames[t] = torch.zeros(
                (B, out_channels, up_H, up_W),
                dtype=x.dtype, device=device
            )
            # We'll store scalar weights per frame, but we need it to broadcast over (B, outC, upH, upW)
            weight_frames[t] = torch.zeros(
                (B, 1, 1, 1),
                dtype=x.dtype, device=device
            )

        # 3) Compute chunk boundaries
        chunk_starts = list(range(0, T, chunk_size))

        for i, start_t in enumerate(chunk_starts):
            end_t = start_t + chunk_size + overlap_size
            if end_t > T:
                end_t = T

            # Extract chunk
            x_chunk = x[:, :, start_t:end_t, :, :]  # shape: (B,C,chunk_len,H,W)
            chunk_len = x_chunk.shape[2]

            # Forward pass for this chunk
            out_chunk = self.forward_3d(x_chunk)  # (B, out_channels, chunk_len, up_H, up_W)

            # 4) Build a per-frame weight vector for blending
            chunk_weight = torch.ones(chunk_len, device=device, dtype=x.dtype)

            # Overlap region at the start if i>0
            overlap = min(overlap_size, chunk_len)
            if i > 0 and overlap > 0:
                # Fade in: linearly ramp from 0 -> 1 across the first `overlap` frames
                for t_idx in range(overlap):
                    alpha = float(t_idx + 1) / overlap  # e.g. 1/overlap, 2/overlap, ...
                    chunk_weight[t_idx] = alpha

            # Overlap region at the end if there's another chunk after this
            if (start_t + chunk_size) < T:
                for t_idx in range(chunk_len - overlap, chunk_len):
                    idx_from_end = (chunk_len - 1) - t_idx
                    alpha = float(idx_from_end) / overlap
                    chunk_weight[t_idx] = alpha

            # 5) Accumulate the results frame-by-frame
            #    out_chunk is shape (B, outC, chunk_len, up_H, up_W)
            #    chunk_weight is shape (chunk_len,)
            # We'll multiply out_chunk[:, :, frame_offset] by chunk_weight[frame_offset]
            # and add it to out_frames[t_global].
            for frame_offset in range(chunk_len):
                t_global = start_t + frame_offset
                w = chunk_weight[frame_offset]  # scalar

                # Add to the frame accumulator
                out_frames[t_global] += out_chunk[:, :, frame_offset] * w
                # Accumulate weight
                weight_frames[t_global] += w

        # 6) Normalize each frame by its total weight
        final_frames = []
        for t in range(T):
            w = weight_frames[t]
            # Avoid division by zero
            w_clamped = torch.clamp(w, min=1e-8)
            # out_frames[t] shape: (B, out_channels, up_H, up_W)
            # w shape: (B, 1, 1, 1) => broadcasts
            normalized_frame = out_frames[t] / w_clamped
            final_frames.append(normalized_frame.unsqueeze(2))
            # unsqueeze(2) to insert time dimension => (B, outC, 1, up_H, up_W)

        # 7) Stack along the time dimension
        #    final shape => (B, outC, T, up_H, up_W)
        out_video = torch.cat(final_frames, dim=2)

        return out_video
    
    def _set_gradient_checkpointing(self, module, value=False):
        module.gradient_checkpointing = value

    def forward(self, x, chunk_size=32, overlap_size=8):
        if x.shape[2] <= chunk_size + overlap_size:
            return self.forward_3d(x)
        else:
            return self.forward_in_chunks(x, chunk_size=chunk_size, overlap_size=overlap_size)

########################

class ResBlock2p1D(nn.Module):
    """
    A simple residual block using (2+1)D convolution.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv2p1D(channels, channels, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2 = Conv2p1D(channels, channels, kernel_size=(3,3,3), padding=(1,1,1))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual

class DeformAlign2p1D(nn.Module):
    """
    Placeholder for a (2+1)D deformable alignment or attention-based alignment.
    In EDVR/BasicVSR++, we often see DCN-based alignment or flow-based alignment.
    This simplified version could be replaced with actual DCN if you have it implemented.
    """
    def __init__(self, channels):
        super().__init__()
        self.resblock = ResBlock2p1D(channels)
        # Could add offset conv layers if you want real DCN. Here, it's just a stub.

    def forward(self, x, ref=None):
        # x shape: (B, C, T, H, W)
        # For alignment, we'd estimate offsets or do attention. This is just a toy.
        return self.resblock(x)

# -------------------------------------------------------------------------
# BasicVSR++-inspired recurrent structure (Forward & Backward Propagation)
# -------------------------------------------------------------------------
class BasicVSRpp2p1DNet(ModelMixin, ConfigMixin):
    """
    A BasicVSR++-style approach using (2+1)D blocks:
      1) Feature extraction
      2) Forward propagation
      3) Backward propagation
      4) Fusion
      5) Upsampling
    Also uses chunking in the forward pass to handle large T.
    """
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(
        self, 
        in_channels=1, 
        out_channels=1, 
        base_channels=64,
        num_resblocks=8,
        chunk_size=64,
        overlap=8
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Feature extractor (downsampling 3 times for 40->5 in spatial dims)
        # We'll do a small U-like structure or just repeated convs?
        # For clarity, let's do:
        self.fe_down1 = nn.Conv3d(in_channels, base_channels, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.fe_down2 = nn.Conv3d(base_channels, base_channels*2, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.fe_down3 = nn.Conv3d(base_channels*2, base_channels*4, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.fe_conv = nn.Sequential(*[ResBlock2p1D(base_channels*4) for _ in range(2)])

        # Alignment module (2+1)D-based or DCN-based
        self.align = DeformAlign2p1D(base_channels*4)

        # Propagation blocks (shared by forward/backward)
        self.propagate = nn.Sequential(*[ResBlock2p1D(base_channels*4) for _ in range(num_resblocks)])

        # Fusion after forward & backward
        self.fusion = nn.Sequential(
            nn.Conv3d(base_channels*4*2, base_channels*4, kernel_size=1),
            nn.ReLU(inplace=True),
            *[ResBlock2p1D(base_channels*4) for _ in range(2)]
        )

        # Upsampling path (reverse of the downsample)
        # We'll do a single 4× upsample at the very end for 40->160, but 
        # we also need to reverse the factor-of-8 downsample in the feature extractor.
        # Actually, we did 2×2×2 in each step => total 8× down in H/W. 
        # So let's do 3 transposed conv steps, then final 2× to get 40->160 total.
        self.up3 = nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), output_padding=(0,0,0))
        self.up2 = nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), output_padding=(0,0,0))
        self.up1 = nn.ConvTranspose3d(base_channels, base_channels, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), output_padding=(0,0,0))
        self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward_chunk(self, x):
        """
        x shape: (B, in_channels=1, T_chunk, 40, 40)
        Returns: (B, out_channels=1, T_chunk, 160, 160)
        """

        # 1) Feature Extraction
        f1 = self.fe_down1(x)  # (B, base, T_chunk, 20, 20)
        f2 = self.fe_down2(f1) # (B, 2*base, T_chunk, 10, 10)
        f3 = self.fe_down3(f2) # (B, 4*base, T_chunk, 5, 5)
        feat = self.fe_conv(f3)# (B, 4*base, T_chunk, 5, 5)

        # 2) Recurrent Forward Pass
        B, C, Tc, H, W = feat.shape
        hidden_f = torch.zeros_like(feat[:, :, 0:1])  # hidden state for forward
        forward_feats = []
        for t in range(Tc):
            # Align current frame with previous hidden (toy alignment)
            ft = self.align(feat[:, :, t:t+1], ref=hidden_f)
            # Combine with hidden state
            ft = ft + hidden_f
            # Propagation
            ft = self.propagate(ft)
            # Update hidden
            hidden_f = ft
            forward_feats.append(ft)

        forward_feats = torch.cat(forward_feats, dim=2)  # (B, C, Tc, H, W)

        # 3) Recurrent Backward Pass
        hidden_b = torch.zeros_like(feat[:, :, 0:1])
        backward_feats = [None]*Tc
        for t in range(Tc-1, -1, -1):
            ft = self.align(feat[:, :, t:t+1], ref=hidden_b)
            ft = ft + hidden_b
            ft = self.propagate(ft)
            hidden_b = ft
            backward_feats[t] = ft
        backward_feats = torch.cat(backward_feats, dim=2)  # (B, C, Tc, H, W)

        # 4) Fuse forward & backward
        fb = torch.cat([forward_feats, backward_feats], dim=1)  # (B, 2*C, Tc, H, W)
        fused = self.fusion(fb)  # (B, C, Tc, H, W)

        # 5) Upsampling
        #    Reverse the downsampling steps
        up_3 = self.up3(fused)      # => (B, 2*base, Tc, 10, 10)
        up_2 = self.up2(up_3)       # => (B, base, Tc, 20, 20)
        up_1 = self.up1(up_2)       # => (B, base, Tc, 40, 40)
        # final 4× upsample from 40->160
        out_spatial = F.interpolate(up_1, scale_factor=(1,4,4), mode='trilinear', align_corners=False)
        out = self.final_conv(out_spatial)  # => (B, out_channels, Tc, 160, 160)

        return out
    
    def _set_gradient_checkpointing(self, module, value=False):
        module.gradient_checkpointing = value

    def forward(self, x):
        """
        x shape: (B, in_channels=1, T=224, 40, 40)
        We'll chunk along T dimension with overlap.
        """
        B, C, T, H, W = x.shape
        stride = self.chunk_size - self.overlap
        starts = list(range(0, T, stride))
        if starts[-1] + self.chunk_size < T:
            starts.append(T - self.chunk_size)

        # We'll accumulate in out_acc/out_count for blending overlaps
        out_acc = x.new_zeros((B, 1, T, 160, 160))
        out_count = x.new_zeros((B, 1, T, 160, 160))

        for start_t in starts:
            end_t = min(start_t + self.chunk_size, T)
            x_chunk = x[:, :, start_t:end_t, :, :]  # shape: (B,1,Tc,40,40)
            out_chunk = self.forward_chunk(x_chunk) # => (B,1,Tc,160,160)

            out_acc[:, :, start_t:end_t, :, :] += out_chunk
            out_count[:, :, start_t:end_t, :, :] += 1

        out_count = torch.clamp_min(out_count, 1.0)
        out = out_acc / out_count
        return out

############################
    

class SRUNet2p1D(ModelMixin, ConfigMixin):
    """
    A deeper (2+1)D U-Net with attention blocks to handle more capacity.
    - Input shape: (B, in_channels, T=224, H=40, W=40)
    - Output shape: (B, out_channels, T=224, H=160, W=160)
    """
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 base_channels=64):
        super().__init__()

        # Down: 4 levels
        # Optionally add attention at certain stages, e.g. after the second or third level
        self.down1 = DownBlockAttn(in_channels, base_channels, use_attn=False)           # 40->20
        self.down2 = DownBlockAttn(base_channels, base_channels*2, use_attn=False)       # 20->10
        self.down3 = DownBlockAttn(base_channels*2, base_channels*4, use_attn=True)     # 10->5
        #self.down4 = DownBlockAttn(base_channels*4, base_channels*8, use_attn=True)    # 5->2 or 3? (Note: 5->2 if stride=2, int division)

        # Bottleneck: apply heavy convs + attention
        self.bottleneck = nn.Sequential(
            Conv2p1D(base_channels*4, base_channels*4),
            SimpleSelfAttention3D(base_channels*4),
            Conv2p1D(base_channels*4, base_channels*4),
        )

        # Up: 4 levels
        # Combine skip connections
        #self.up4 = UpBlockAttn(base_channels*8 + base_channels*8, base_channels*4, use_attn=True)
        self.up3 = UpBlockAttn(base_channels*4 + base_channels*4, base_channels*2, use_attn=True)
        self.up2 = UpBlockAttn(base_channels*2 + base_channels*2, base_channels, use_attn=False)
        self.up1 = UpBlockAttn(base_channels + base_channels, base_channels, use_attn=False)

        self.upsample_tconv1 = nn.ConvTranspose3d(
            in_channels=base_channels,  # or whatever your up_1 channels are
            out_channels=base_channels, 
            kernel_size=(1,4,4), 
            stride=(1,2,2), 
            padding=(0,1,1)
        )
        self.post_block1 = ResBlock2p1D(base_channels)
        self.upsample_tconv2 = nn.ConvTranspose3d(
            in_channels=base_channels,
            out_channels=out_channels,
            kernel_size=(1,4,4),
            stride=(1,2,2),
            padding=(0,1,1)
        )
        self.post_block2 = ResBlock2p1D(out_channels)       

    def forward_3d(self, x):
        """
        The main forward pass for the entire volume
        x shape: (B,1,T,H=40,W=40)
        We want: (B,1,T,H=160,W=160)
        """

        skip1, d1 = self.down1(x)   # skip1->(B,64,T,40,40), d1->(B,64,T,20,20)
        skip2, d2 = self.down2(d1)  # skip2->(B,128,T,20,20), d2->(B,128,T,10,10)
        skip3, d3 = self.down3(d2)  # skip3->(B,256,T,10,10), d3->(B,256,T,5,5)
        #skip4, d4 = self.down4(d3)  # skip4->(B,512,T,5,5),   d4->(B,512,T,2,2) or (2,3)

        b = self.bottleneck(d3)     # (B,512,T,2,2) or (2,3) depending on shape

        # Up path
        #u4 = self.up4(b, skip4)     # -> (B,256,T,5,5)
        u3 = self.up3(b, skip3)    # -> (B,128,T,10,10)
        u2 = self.up2(u3, skip2)    # -> (B,64,T,20,20)
        u1 = self.up1(u2, skip1)    # -> (B,64,T,40,40)

        # We have 40×40, but need 160×160 => 4× up
        #out = F.interpolate(u1, scale_factor=(1,4,4), mode='trilinear', align_corners=False)
        x = self.upsample_tconv1(u1)
        x = F.relu(x, inplace=True)
        x = self.post_block1(x)  # learned refinement at 80x80
        # Second transposed conv: 80 -> 160
        x = self.upsample_tconv2(x)  
        out = self.post_block2(x)  # final learned refinement at 160x160

        return out

    def forward_in_chunks(self, x, chunk_size=32, overlap_size=8):
        """
        Process x along the T dimension in overlapping chunks, blending them
        in a fully differentiable way so gradients can flow back through each chunk.

        x shape: (B, C, T, H, W)
        returns shape: (B, out_channels, T, 4H, 4W)
        """
        device = x.device
        B, C, T, H, W = x.shape

        # 1) Figure out output shape by running a "dummy" chunk.
        #    Remove the no_grad() so that PyTorch won't break the graph later,
        #    but typically this dummy pass won't consume much memory if chunk_size is small.
        dummy_chunk_len = min(chunk_size, T)  # in case T < chunk_size
        dummy_out = self.forward_3d(x[:, :, :dummy_chunk_len, :, :])
        out_channels = dummy_out.shape[1]
        up_H = dummy_out.shape[3]  # if you're doing 4× upsampling: up_H = 4*H
        up_W = dummy_out.shape[4]  # up_W = 4*W

        # 2) We'll accumulate per-frame outputs in Python lists, one entry for each of T frames.
        #    Each entry is a tensor of shape (B, out_channels, up_H, up_W).
        #    We'll also keep a "weight" tensor for each frame to handle overlapping blends.
        out_frames = [None] * T
        weight_frames = [None] * T

        # Initialize each frame's accumulator
        for t in range(T):
            out_frames[t] = torch.zeros(
                (B, out_channels, up_H, up_W),
                dtype=x.dtype, device=device
            )
            # We'll store scalar weights per frame, but we need it to broadcast over (B, outC, upH, upW)
            weight_frames[t] = torch.zeros(
                (B, 1, 1, 1),
                dtype=x.dtype, device=device
            )

        # 3) Compute chunk boundaries
        chunk_starts = list(range(0, T, chunk_size))

        for i, start_t in enumerate(chunk_starts):
            end_t = start_t + chunk_size + overlap_size
            if end_t > T:
                end_t = T

            # Extract chunk
            x_chunk = x[:, :, start_t:end_t, :, :]  # shape: (B,C,chunk_len,H,W)
            chunk_len = x_chunk.shape[2]

            # Forward pass for this chunk
            out_chunk = self.forward_3d(x_chunk)  # (B, out_channels, chunk_len, up_H, up_W)

            # 4) Build a per-frame weight vector for blending
            chunk_weight = torch.ones(chunk_len, device=device, dtype=x.dtype)

            # Overlap region at the start if i>0
            overlap = min(overlap_size, chunk_len)
            if i > 0 and overlap > 0:
                # Fade in: linearly ramp from 0 -> 1 across the first `overlap` frames
                for t_idx in range(overlap):
                    alpha = float(t_idx + 1) / overlap  # e.g. 1/overlap, 2/overlap, ...
                    chunk_weight[t_idx] = alpha

            # Overlap region at the end if there's another chunk after this
            if (start_t + chunk_size) < T:
                for t_idx in range(chunk_len - overlap, chunk_len):
                    idx_from_end = (chunk_len - 1) - t_idx
                    alpha = float(idx_from_end) / overlap
                    chunk_weight[t_idx] = alpha

            # 5) Accumulate the results frame-by-frame
            #    out_chunk is shape (B, outC, chunk_len, up_H, up_W)
            #    chunk_weight is shape (chunk_len,)
            # We'll multiply out_chunk[:, :, frame_offset] by chunk_weight[frame_offset]
            # and add it to out_frames[t_global].
            for frame_offset in range(chunk_len):
                t_global = start_t + frame_offset
                w = chunk_weight[frame_offset]  # scalar

                # Add to the frame accumulator
                out_frames[t_global] += out_chunk[:, :, frame_offset] * w
                # Accumulate weight
                weight_frames[t_global] += w

        # 6) Normalize each frame by its total weight
        final_frames = []
        for t in range(T):
            w = weight_frames[t]
            # Avoid division by zero
            w_clamped = torch.clamp(w, min=1e-8)
            # out_frames[t] shape: (B, out_channels, up_H, up_W)
            # w shape: (B, 1, 1, 1) => broadcasts
            normalized_frame = out_frames[t] / w_clamped
            final_frames.append(normalized_frame.unsqueeze(2))
            # unsqueeze(2) to insert time dimension => (B, outC, 1, up_H, up_W)

        # 7) Stack along the time dimension
        #    final shape => (B, outC, T, up_H, up_W)
        out_video = torch.cat(final_frames, dim=2)

        return out_video
    
    def _set_gradient_checkpointing(self, module, value=False):
        module.gradient_checkpointing = value

    def forward(self, x, chunk_size=32, overlap_size=8):
        if x.shape[2] <= chunk_size + overlap_size:
            return self.forward_3d(x)
        else:
            return self.forward_in_chunks(x, chunk_size=chunk_size, overlap_size=overlap_size)


class CNN_3D(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(self, in_channels):
        super(CNN_3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 32, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.MaxPool3d(kernel_size=(2,2,2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.MaxPool3d(kernel_size=(2,2,2))
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv3d(64, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.MaxPool3d(2)
        )

        self.conv6 = nn.Sequential(
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.MaxPool3d(2)
        )

        self.conv7 = nn.Sequential(
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96)
        )
        
        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(768, 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def _set_gradient_checkpointing(self, module, value=False):
        module.gradient_checkpointing = value
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x