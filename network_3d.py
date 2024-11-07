# network_3d.py
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from module_3d import DownSample3D, ResBlock3D, Swish3D, TimeEmbedding3D, UpSample3D
from torch.nn import init

class UNet3D(nn.Module):
    def __init__(
        self, 
        T=1000, 
        voxel_resolution=64, 
        ch=128, 
        ch_mult=[1,2,2,2], 
        attn=[1], 
        num_res_blocks=4, 
        dropout=0.1, 
        use_cfg=False, 
        cfg_dropout=0.1, 
        num_classes=None
    ):
        super().__init__()
        self.voxel_resolution = voxel_resolution
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding3D(tdim)

        # Classifier-free guidance
        self.use_cfg = use_cfg
        self.cfg_dropout = cfg_dropout
        if use_cfg:
            assert num_classes is not None
            cdim = tdim
            self.class_embedding = nn.Embedding(num_classes+1, cdim)

        self.head = nn.Conv3d(1, ch, kernel_size=3, stride=1, padding=1)  # Assuming single channel voxel grid
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when downsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock3D(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample3D(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock3D(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock3D(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock3D(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample3D(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish3D(),
            nn.Conv3d(now_ch, 1, 3, stride=1, padding=1)  # Output single channel
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, timestep, class_label=None):
        # Timestep embedding
        temb = self.time_embedding(timestep)

        if self.use_cfg and class_label is not None:
            if self.training:
                assert not torch.any(class_label == 0)  # 0 for null.
                # Random null conditioning
                mask = torch.rand_like(class_label, dtype=torch.float32) > self.cfg_dropout
                class_label = class_label * mask.long()

            # Class conditioning
            class_emb = self.class_embedding(class_label)
            temb = temb + class_emb

        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock3D):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h
