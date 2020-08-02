from typing import Dict, Union
import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import math
from gym_loop.agents.layers.noisy_layer import NoisyLinear
from torch.distributions.categorical import Categorical


class NoisyActorCritic(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(NoisyActorCritic, self).__init__()
        self.out_dim = out_dim

        # set common feature layer
        self.feature_layer = nn.Sequential(NoisyLinear(in_dim, 128), nn.ReLU())

        # set advantage layer
        self.policy_hidden_layer = nn.Linear(128, 128)
        self.policy_layer = nn.Linear(128, out_dim)

        # set value layer
        self.value_hidden_layer = nn.Linear(128, 128)
        self.value_layer = nn.Linear(128, 1)

    def act(self, x: np.ndarray) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        outp = self(x)
        outp["action"] = Categorical(outp["action_distribution"]).sample()
        return outp

    def forward(self, x: np.ndarray) -> Dict[str, torch.Tensor]:
        """Forward method implementation."""
        x = torch.FloatTensor(x)
        feature = self.feature_layer(x)
        policy_hid = F.relu(self.policy_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        actions = F.softmax(self.policy_layer(feature), dim=-1)
        value = self.value_layer(feature)

        return {"action_distribution": actions, "values": value}

    def reset_noise(self):
        for module in self.modules():
            if hasattr(module, "reset_noise") and not module is self:
                module.reset_noise()
