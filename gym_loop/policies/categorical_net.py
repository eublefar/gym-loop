from typing import Dict, List, Tuple
import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import math
from gym_loop.agents.layers.noisy_layer import NoisyLinear
from torch.distributions.categorical import Categorical
from .base_policy import BasePolicy


class CategoricalNet(BasePolicy):
    def __init__(
        self, in_dim: int, out_dim: int, atom_size: int, support: torch.Tensor
    ):
        """Initialization."""
        super(CategoricalNet, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.feature_layer = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU())

        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, atom_size)

    def forward(self, x: np.ndarray) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.FloatTensor(x)
        dist = self.distribution(x)
        return {"q_values": torch.sum(dist * self.support, dim=2)}

    def distribution(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(-1, self.out_dim, self.atom_size)
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist

    def reset_noise(self):
        for module in self.value_layer.modules():
            if hasattr(module, "reset_noise"):
                module.reset_noise()
        for module in self.advantage_layer.modules():
            if hasattr(module, "reset_noise"):
                module.reset_noise()
