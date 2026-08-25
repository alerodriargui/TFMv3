"""Pérdida perceptual con MAE ViT-L congelado (multi-escala)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def _cosine_distance(
    feat_a: torch.Tensor, feat_b: torch.Tensor
) -> torch.Tensor:
    sim = F.cosine_similarity(feat_a, feat_b, dim=1)
    return 1.0 - sim.unsqueeze(1)


class PerceptualLoss(nn.Module):
    """Pérdida perceptual basada en features de MAE ViT-L congelado.

    Extrae features de capas intermedias del MAE ViT-L y calcula la
    distancia coseno entre las features de la imagen de entrada y la
    reconstrucción. Soporta multi-escala con diferentes patch sizes.
    """

    def __init__(
        self,
        model_name: str = "vit_large_patch16_224.mae",
        layers: tuple[int, ...] = (15, 19),
        patch_sizes: tuple[int, ...] = (32, 56),
        img_size: int = 224,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.patch_sizes = patch_sizes
        self.img_size = img_size
        self.models = nn.ModuleList()
        for ps in patch_sizes:
            n_patches = img_size // ps
            model = timm.create_model(
                model_name,
                pretrained=True,
                patch_size=ps,
                img_size=img_size,
            )
            model.eval()
            model.requires_grad_(False)
            self.models.append(model)

    @torch.no_grad()
    def extract_features(
        self, images: torch.Tensor, model_idx: int
    ) -> list[torch.Tensor]:
        model = self.models[model_idx]
        hstates = model.forward_intermediates(
            images,
            output_fmt="NLC",
            intermediates_only=True,
            return_prefix_tokens=True,
            norm=True,
        )
        features = []
        for h_patch, h_cls in hstates:
            combined = torch.cat([h_cls, h_patch], dim=1)
            n_reg = combined.shape[1] - (self.img_size // self.patch_sizes[model_idx]) ** 2
            features.append(combined[:, n_reg:])
        return features

    def _select_layers(
        self, all_features: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        selected = []
        for idx in self.layers:
            actual = idx
            if actual < 0:
                actual = len(all_features) + actual
            if actual < len(all_features):
                selected.append(all_features[actual])
            else:
                selected.append(all_features[-1])
        return selected

    def forward(
        self, images: torch.Tensor, reconstructions: torch.Tensor
    ) -> torch.Tensor:
        all_loss_maps = []
        for model_idx in range(len(self.models)):
            feats_in = self._select_layers(self.extract_features(images, model_idx))
            feats_re = self._select_layers(
                self.extract_features(reconstructions, model_idx)
            )
            losses = []
            for f_in, f_re in zip(feats_in, feats_re):
                d = _cosine_distance(f_in.permute(0, 2, 1), f_re.permute(0, 2, 1))
                losses.append(d.mean(dim=-1, keepdim=True))
            loss_stack = torch.cat(losses, dim=1)
            H = W = int(loss_stack.shape[-1] ** 0.5)
            loss_map = loss_stack.reshape(
                loss_stack.shape[0], loss_stack.shape[1], H, W
            )
            loss_map = F.interpolate(
                loss_map,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
            all_loss_maps.append(loss_map)

        product = all_loss_maps[0]
        for lm in all_loss_maps[1:]:
            product = product * lm
        return product

    def loss_and_maps(
        self, images: torch.Tensor, reconstructions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss_maps = self.forward(images, reconstructions)
        loss_scalar = loss_maps.mean()
        return loss_scalar, loss_maps
