"""Denoising Autoencoder for brain anomaly detection.

Based on Kascenas et al. (MIDL 2022):
  U-Net style DAE with three downsampling stages and skip connections.
  Train: add coarse noise, then reconstruct the clean image.
  Score: use reconstruction error as the anomaly map.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Bloque de convolución
class ConvBlock(nn.Module):
    # Inicializa el bloque de convolución con capas de convolución, activación SiLU y normalización por lotes.
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.GroupNorm(8, out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.GroupNorm(8, out_ch),
        )
    # Realiza la pasada hacia adelante del bloque de convolución.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

# Autoencoder de eliminación de ruido (DAE) 
class DAE(nn.Module):
    # Inicializa el DAE con canales de entrada y un número base de canales, construyendo la arquitectura U-Net con bloques de convolución, capas de pooling y capas de upsampling.
    def __init__(self, in_channels: int = 1, base_ch: int = 64) -> None:
        super().__init__()
        c = base_ch

        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)

        self.pool = nn.AvgPool2d(2)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c * 8, c * 4, 3, padding=1, bias=False),
        )
        self.dec3 = ConvBlock(c * 8, c * 4)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c * 4, c * 2, 3, padding=1, bias=False),
        )
        self.dec2 = ConvBlock(c * 4, c * 2)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c * 2, c, 3, padding=1, bias=False),
        )
        self.dec1 = ConvBlock(c * 2, c)

        self.head = nn.Conv2d(c, in_channels, 1)

    # Realiza la pasada hacia adelante del DAE, pasando la entrada a través de los bloques de codificación, aplicando pooling, luego pasando a través de los bloques de decodificación con concatenación de las características correspondientes de la codificación y finalmente produciendo la salida reconstruida.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)
