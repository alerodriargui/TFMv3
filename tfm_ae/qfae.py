"""Q-Former Autoencoder para detección de anomalías.

Arquitectura basada en Dalmonte et al. (WACV 2026):
  DINOv2 ViT-L (congelado) → Q-Former bottleneck → Transformer decoder
  Loss: perceptual coseno (MAE ViT-L, multi-escala)
"""

from __future__ import annotations

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])

    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    emb_h = np.einsum("m,d->md", grid[0].reshape(-1), omega)
    emb_w = np.einsum("m,d->md", grid[1].reshape(-1), omega)
    emb = np.concatenate(
        [np.sin(emb_h), np.cos(emb_h), np.sin(emb_w), np.cos(emb_w)], axis=1
    )
    return torch.from_numpy(emb).float().unsqueeze(0)


def _unpatchify(
    x: torch.Tensor, patch_size: int, n_channels: int
) -> torch.Tensor:
    h = w = int(x.shape[1] ** 0.5)
    p = patch_size
    x = x.reshape(x.shape[0], h, w, p, p, n_channels)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(x.shape[0], n_channels, h * p, w * p)


class DINOv2Encoder(nn.Module):
    """Encoder DINOv2 ViT congelado que extrae features intermedias."""

    def __init__(
        self,
        model_name: str = "vit_large_patch14_reg4_dinov2.lvd142m",
        img_size: int = 224,
        use_hidden_state: tuple[int, ...] = (-2, -4),
        out_dim: int = 768,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.use_hidden_state = use_hidden_state
        self.out_dim = out_dim

        self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval()
        self.model.requires_grad_(False)

        self.patch_size = 14
        self.n_patches = img_size // self.patch_size

        embed_dim = self.model.embed_dim
        self.projection = nn.Linear(embed_dim, out_dim)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hstates = self.model.forward_intermediates(
            x,
            output_fmt="NLC",
            intermediates_only=True,
            return_prefix_tokens=True,
            norm=True,
        )
        selected = []
        for idx in self.use_hidden_state:
            actual = idx
            if actual < 0:
                actual = len(hstates) + actual
            h_patch, h_cls = hstates[actual]
            combined = torch.cat([h_cls, h_patch], dim=1)
            n_reg = combined.shape[1] - self.n_patches**2
            selected.append(combined[:, n_reg:])

        features = torch.cat(selected, dim=1)
        return self.projection(features)


class QFormerBottleneck(nn.Module):
    """Q-Former: bottleneck con learnable queries y cross-attention."""

    def __init__(
        self,
        dim: int = 768,
        n_queries: int = 784,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.01,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(1, n_queries, dim))
        nn.init.normal_(self.queries, std=1e-3)

        self.self_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm_sa = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm_ca = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, patch_tokens: torch.Tensor
    ) -> torch.Tensor:
        b = patch_tokens.shape[0]
        x = self.queries.expand(b, -1, -1)

        residual = x
        x = self.norm_sa(x)
        x, _ = self.self_attn(x, x, x)
        x = residual + x

        residual = x
        x_q = self.norm_ca(x)
        kv = self.norm_kv(patch_tokens)
        x_kv, _ = self.cross_attn(x_q, kv, kv)
        x = residual + x_kv

        residual = x
        x = residual + self.ffn(x)

        return x


class TransformerDecoder(nn.Module):
    """Decoder Transformer estilo MAE."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 8,
        in_dim: int = 768,
        decoder_dim: int = 768,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_channels: int = 1,
        n_reg_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = img_size // patch_size
        self.out_channels = out_channels
        self.n_add_tokens = 1 + n_reg_tokens

        self.decoder_embed = nn.Linear(in_dim, decoder_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.reg_tokens = nn.Parameter(torch.zeros(1, n_reg_tokens, decoder_dim))
        nn.init.normal_(self.cls_token, std=1e-3)
        nn.init.normal_(self.reg_tokens, std=1e-3)

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches**2, decoder_dim), requires_grad=False
        )
        pos = _get_2d_sincos_pos_embed(decoder_dim, self.n_patches)
        self.decoder_pos_embed.data.copy_(pos)

        self.decoder_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=decoder_dim,
                    nhead=num_heads,
                    dim_feedforward=int(decoder_dim * mlp_ratio),
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_prediction = nn.Linear(
            decoder_dim, patch_size**2 * out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed

        cls = self.cls_token.expand(x.shape[0], -1, -1)
        reg = self.reg_tokens.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, reg, x], dim=1)

        for block in self.decoder_blocks:
            x = block(x)

        x = self.decoder_norm(x)
        x = x[:, self.n_add_tokens :]
        x = self.decoder_prediction(x)
        return _unpatchify(x, self.patch_size, self.out_channels)


class QFAE(nn.Module):
    """Q-Former Autoencoder completo."""

    def __init__(
        self,
        encoder_name: str = "vit_large_patch14_reg4_dinov2.lvd142m",
        img_size: int = 224,
        use_hidden_state: tuple[int, ...] = (-2, -4),
        encoder_dim: int = 768,
        junction_dim: int = 768,
        junction_n_queries: int = 784,
        junction_heads: int = 8,
        junction_mlp_ratio: float = 4.0,
        decoder_dim: int = 768,
        decoder_depth: int = 6,
        decoder_heads: int = 12,
        decoder_mlp_ratio: float = 4.0,
        decoder_patch_size: int = 8,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        self.img_size = img_size

        self.encoder = DINOv2Encoder(
            model_name=encoder_name,
            img_size=img_size,
            use_hidden_state=use_hidden_state,
            out_dim=encoder_dim,
        )

        self.qformer = QFormerBottleneck(
            dim=junction_dim,
            n_queries=junction_n_queries,
            heads=junction_heads,
            mlp_ratio=junction_mlp_ratio,
        )

        self.decoder = TransformerDecoder(
            img_size=img_size,
            patch_size=decoder_patch_size,
            in_dim=junction_dim,
            decoder_dim=decoder_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
            mlp_ratio=decoder_mlp_ratio,
            out_channels=out_channels,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] == 1:
            images_3ch = images.repeat(1, 3, 1, 1)
        else:
            images_3ch = images
        features = self.encoder(images_3ch)
        latent = self.qformer(features)
        return self.decoder(latent)


def build_qfae_model(
    encoder_name: str = "vit_large_patch14_reg4_dinov2.lvd142m",
    img_size: int = 224,
    junction_dim: int = 768,
    junction_n_queries: int = 784,
    junction_heads: int = 8,
    decoder_dim: int = 768,
    decoder_depth: int = 6,
    decoder_heads: int = 12,
) -> QFAE:
    return QFAE(
        encoder_name=encoder_name,
        img_size=img_size,
        junction_dim=junction_dim,
        junction_n_queries=junction_n_queries,
        junction_heads=junction_heads,
        decoder_dim=decoder_dim,
        decoder_depth=decoder_depth,
        decoder_heads=decoder_heads,
    )
