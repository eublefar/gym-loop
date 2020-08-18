from typing import Dict, Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gym
from gym_loop.agents.layers.noisy_layer import NoisyLinear
from torch.distributions import Categorical, MultivariateNormal
from .base_policy import BasePolicy


class DiscreteNoisyActorCritic(BasePolicy, nn.Module):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        """Initialization."""
        nn.Module.__init__(self)

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128), nn.ReLU()
        )

        # set advantage layer
        self.policy_hidden_layer = nn.Linear(128, 128)
        self.policy_layer = nn.Linear(128, action_space.n)

        # set value layer
        self.value_hidden_layer = nn.Linear(128, 128)
        self.value_layer = nn.Linear(128, 1)

    def forward(self, x: np.ndarray) -> Dict[str, torch.Tensor]:
        """Forward method implementation."""
        x = torch.FloatTensor(x)
        feature = self.feature_layer(x)
        policy_hid = F.relu(self.policy_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        actions = F.softmax(self.policy_layer(policy_hid), dim=-1)
        value = self.value_layer(val_hid)
        distributions = Categorical(actions)

        return {"action_distribution": distributions, "values": value}

    def __call__(self, *args):
        return nn.Module.__call__(self, *args)

    def reset_noise(self):
        for module in self.modules():
            if hasattr(module, "reset_noise") and module is not self:
                module.reset_noise()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class GaussianNoisyActorCritic(BasePolicy, nn.Module):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        """Initialization."""
        nn.Module.__init__(self)

        # set common feature layer
        self.feature_layer = nn.Sequential(
            NoisyLinear(observation_space.shape[0], 128), nn.ReLU()
        )

        if len(action_space.shape) > 1:
            raise ValueError("Complex shapes are not supported for actions")

        # set advantage layer
        self.policy_hidden_layer = nn.Linear(128, 128)
        self.policy_layer = nn.Linear(128, action_space.shape[0] * 2)

        self.action_high = torch.FloatTensor(action_space.high)
        self.action_low = torch.FloatTensor(action_space.low)
        self.action_scale = self.action_high - self.action_low

        # set value layer
        self.value_hidden_layer = nn.Linear(128, 128)
        self.value_layer = nn.Linear(128, 1)

    def forward(self, x: np.ndarray) -> Dict[str, torch.Tensor]:
        """Forward method implementation."""
        x = torch.FloatTensor(x)
        feature = self.feature_layer(x)

        policy_hid = F.relu(self.policy_hidden_layer(feature))

        val_hid = F.relu(self.value_hidden_layer(feature))

        policy_out = self.policy_layer(policy_hid)

        action_loc, action_scale = policy_out.chunk(2, dim=-1)
        action_loc = torch.sigmoid(action_loc)
        action_loc = self._scale(action_loc)

        action_scale = self.action_scale * F.relu(action_scale) + 1e-8

        value = self.value_layer(val_hid)
        distributions = MultivariateNormal(action_loc, action_scale.diag_embed())
        return {"action_distribution": distributions, "values": value}

    def _scale(self, actions):
        return self.action_low + actions * self.action_scale

    def __call__(self, *args):
        return nn.Module.__call__(self, *args)

    def reset_noise(self):
        for module in self.modules():
            if hasattr(module, "reset_noise") and module is not self:
                module.reset_noise()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))

