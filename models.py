"""Small convolutional models used in the experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def encoder_blocks() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 16, 4, 2, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(16, 32, 4, 2, 1),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(32, 64, 4, 2, 1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 128, 4, 2, 1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
    )


def decoder_blocks() -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(128, 64, 4, 2, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 32, 4, 2, 1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(32, 16, 4, 2, 1),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(16, 1, 4, 2, 1),
        nn.Sigmoid(),
    )


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        del latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images))


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.features = encoder_blocks()
        self.mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.logvar = nn.Linear(128 * 4 * 4, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = decoder_blocks()

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(images).flatten(1)
        return self.mu(features), self.logvar(features).clamp(-12, 12)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        features = self.from_latent(latent).view(-1, 128, 4, 4)
        return self.decoder(features)

    def forward(
        self, images: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(images)
        latent = self.reparameterize(mu, logvar) if sample else mu
        return self.decode(latent), mu, logvar


class LatentEncoder(nn.Module):
    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.features = encoder_blocks()
        self.to_latent = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.features(images).flatten(1))


class GANomalyGenerator(nn.Module):
    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.encoder1 = LatentEncoder(latent_dim)
        self.from_latent = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = decoder_blocks()
        self.encoder2 = LatentEncoder(latent_dim)

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_in = self.encoder1(images)
        generated = self.decoder(self.from_latent(latent_in).view(-1, 128, 4, 4))
        latent_out = self.encoder2(generated)
        return generated, latent_in, latent_out


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Linear(64 * 8 * 8, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(images)
        logits = self.classifier(features.flatten(1)).squeeze(1)
        return logits, features


@dataclass(frozen=True)
class ModelBundle:
    name: str
    model: nn.Module
    discriminator: nn.Module | None = None


def build_model(name: str, latent_dim: int = 64) -> ModelBundle:
    if name == "ae":
        return ModelBundle(name, ConvAutoencoder(latent_dim))
    if name == "vae":
        return ModelBundle(name, VariationalAutoencoder(latent_dim))
    if name == "ganomaly":
        return ModelBundle(name, GANomalyGenerator(latent_dim), Discriminator())
    raise ValueError(f"Modelo desconocido: {name}")


def per_image_scores(
    bundle: ModelBundle, images: torch.Tensor, vae_beta: float = 1e-4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return method-specific anomaly scores and deterministic reconstructions."""
    if bundle.name == "ae":
        reconstructed = bundle.model(images)
        score = torch.mean(torch.abs(reconstructed - images), dim=(1, 2, 3))
    elif bundle.name == "vae":
        reconstructed, mu, logvar = bundle.model(images, sample=False)
        reconstruction = torch.mean(torch.abs(reconstructed - images), dim=(1, 2, 3))
        kl = -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp(), dim=1)
        score = reconstruction + vae_beta * kl
    else:
        reconstructed, latent_in, latent_out = bundle.model(images)
        score = torch.mean(torch.abs(latent_in - latent_out), dim=1)
    return score, reconstructed


def vae_loss(
    reconstructed: torch.Tensor,
    images: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    reconstruction = F.l1_loss(reconstructed, images)
    kl = -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())
    return reconstruction + beta * kl
