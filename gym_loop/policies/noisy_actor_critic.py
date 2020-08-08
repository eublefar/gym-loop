from typing import Dict, Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gym
from gym_loop.agents.layers.noisy_layer import NoisyLinear
from torch.distributions.categorical import Categorical
from .base_policy import BasePolicy


class NoisyActorCritic(BasePolicy, nn.Module):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        """Initialization."""
        nn.Module.__init__(self)

        # set common feature layer
        self.feature_layer = nn.Sequential(
            NoisyLinear(observation_space.shape[0], 128), nn.ReLU()
        )

        # set advantage layer
        self.policy_hidden_layer = nn.Linear(128, 128)
        self.policy_layer = nn.Linear(128, action_space.n)

        # set value layer
        self.value_hidden_layer = nn.Linear(128, 128)
        self.value_layer = nn.Linear(128, 1)

    def act(self, x: np.ndarray) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        outp = self(x)
        outp["action"] = (
            Categorical(outp["action_distribution"]).sample().detach().numpy()
        )
        return outp

    def batch_act(self, x: np.ndarray) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        outp = self(x)
        outp["actions"] = (
            Categorical(outp["action_distribution"]).sample().detach().numpy()
        )
        return outp

    def forward(self, x: np.ndarray) -> Dict[str, torch.Tensor]:
        """Forward method implementation."""
        x = torch.FloatTensor(x)
        feature = self.feature_layer(x)
        policy_hid = F.relu(self.policy_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        actions = F.softmax(self.policy_layer(policy_hid), dim=-1)
        value = self.value_layer(val_hid)

        return {"action_distribution": actions, "values": value}

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
