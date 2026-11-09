"""Geometry-aware taxel ViTs and the shared cross-modal decoder.

One taxel is one patch (patch size 1). Curved sensors use physical positional
features rather than an invented rectangular image layout.
"""
from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    taxels: int = 120
    real_channels: int = 3
    history: int = 10
    sim_history: int = 1
    width: int = 96
    layers: int = 3
    heads: int = 4
    latent: int = 64
    dropout: float = 0.0
    use_module_force: bool = True

    def __post_init__(self):
        if self.real_channels != 3:
            raise ValueError("ID3 requires three measured force components per taxel")
        if min(self.taxels, self.real_channels, self.history, self.sim_history,
               self.width, self.layers, self.heads, self.latent) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")


class TaxelViT(nn.Module):
    def __init__(self, channels, config, force_channels=0):
        super().__init__()
        d = config.width
        self.value = nn.Linear(channels, d)
        self.position = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.force = nn.Linear(force_channels, d) if force_channels else None
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.force_type = nn.Parameter(torch.randn(1, 1, d) * 0.02) if self.force else None
        layer = nn.TransformerEncoderLayer(d, config.heads, d * 4,
                    dropout=config.dropout, batch_first=True, norm_first=True,
                    activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, config.layers,
                                                enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, config.latent))

    def forward(self, x, geometry, mask, force=None):
        # x: [B,H,N,C]; mask: [B,N]; geometry: [N,6] or [B,N,6].
        b, _, n, _ = x.shape
        tokens = self.value(x.permute(0, 2, 1, 3).reshape(b, n, -1))
        tokens = tokens + self.position(geometry)
        parts = [self.cls.expand(b, -1, -1), tokens]
        valid = [torch.ones(b, 1, dtype=torch.bool, device=x.device), mask]
        if self.force is not None:
            if force is None:
                raise ValueError("Module force is required by this encoder")
            parts.append(self.force(force.flatten(1)).unsqueeze(1) + self.force_type)
            valid.append(torch.ones(b, 1, dtype=torch.bool, device=x.device))
        z = self.transformer(torch.cat(parts, 1),
                             src_key_padding_mask=~torch.cat(valid, 1))[:, 0]
        # Same unit-norm embedding for contrastive training, decoding, and probing.
        return F.normalize(self.output(z), dim=-1)


class TactSpace(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        c = config
        self.encoders = nn.ModuleDict({
            "real": TaxelViT(c.history * c.real_channels, c,
                             c.history * 3 if c.use_module_force else 0),
            "sim": TaxelViT(c.sim_history, c,
                            c.sim_history * 3 if c.use_module_force else 0),
        })
        self.decoder = nn.Sequential(nn.Linear(c.latent, c.width * 2), nn.GELU(),
                                     nn.Linear(c.width * 2, c.width * 2), nn.GELU())
        self.decode_real = nn.Linear(c.width * 2, c.history * c.taxels * c.real_channels)
        self.decode_sim = nn.Linear(c.width * 2, c.sim_history * c.taxels)
        self.decode_forces = nn.ModuleDict({
            domain: nn.Linear(c.width * 2, history * 3)
            for domain, history in (("real", c.history), ("sim", c.sim_history))
        }) if c.use_module_force else nn.ModuleDict()

    def encode(self, batch, domain):
        return self.encoders[domain](batch[domain], batch["geometry"],
                    batch[f"{domain}_mask"], batch.get(f"{domain}_force"))

    def decode(self, z):
        c = self.config
        h = self.decoder(z)
        result = {"real": self.decode_real(h).reshape(-1, c.history, c.taxels, c.real_channels),
                  "sim": self.decode_sim(h).reshape(-1, c.sim_history, c.taxels, 1)}
        for domain, head in self.decode_forces.items():
            history = c.history if domain == "real" else c.sim_history
            result[f"{domain}_force"] = head(h).reshape(-1, history, 3)
        return result

    def forward(self, batch):
        z = {domain: self.encode(batch, domain) for domain in ("real", "sim")}
        return z, {domain: self.decode(value) for domain, value in z.items()}

    def config_dict(self):
        return asdict(self.config)


class FrozenProbe(nn.Module):
    """MLP on frozen embeddings; train() never enables encoder dropout."""
    def __init__(self, encoder: TactSpace, outputs, hidden=128):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()
        self.head = nn.Sequential(nn.Linear(encoder.config.latent, hidden), nn.GELU(),
                                  nn.Linear(hidden, outputs))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, batch, domain):
        with torch.no_grad():
            z = self.encoder.encode(batch, domain)
        return self.head(z)
