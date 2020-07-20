"""Thanks to https://github.com/Curt-Park/rainbow-is-all-you-need"""
from typing import Dict, List, Tuple
from .base_agent import BaseAgent
from .replay_buffers.per_buffer import PrioritizedReplayBuffer
from .replay_buffers.replay_buffer import ReplayBuffer
import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import logging
import math
from .layers.noisy_layer import NoisyLinear
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

logging.basicConfig(level=logging.INFO)


class GMNetwork(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, atom_size: int, support: torch.Tensor
    ):
        """Initialization."""
        super(GMNetwork, self).__init__()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        return torch.sum(dist * self.support, dim=2)

    def dist(self, x: torch.Tensor) -> torch.Tensor:
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


class PytorchRainbowDqn(BaseAgent):
    def __init__(self, **params):
        super().__init__(**params)

        obs_dim = self.observation_space.shape[0]
        action_dim = self.action_space.n

        self.memory = PrioritizedReplayBuffer(
            obs_dim,
            self.memory_size,
            self.batch_size,
            self.buffer_alpha,
            gamma=self.gamma,
        )
        self.memory_n = ReplayBuffer(
            obs_dim,
            self.memory_size,
            self.batch_size,
            n_step=self.n_step,
            gamma=self.gamma,
        )

        self.beta = self.buffer_beta_min
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(
            self.device
        )

        # networks: dqn, dqn_target
        self.dqn = (
            GMNetwork(
                obs_dim, action_dim, atom_size=self.atom_size, support=self.support
            )
            .to(self.device)
            .train()
        )
        self.dqn_target = GMNetwork(
            obs_dim, action_dim, atom_size=self.atom_size, support=self.support
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        self.loss_per_batch = 0
        self.episode_step = 0
        # mode: train / test
        self.is_test = False

    def act(self, state: np.ndarray, global_step: int) -> np.ndarray:
        """Select an action from the input state."""
        selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
        return selected_action.detach().cpu().numpy()

    def memorize(
        self,
        last_ob: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        ob: np.ndarray,
        global_step: int,
    ):
        first_in_seq = self.memory_n.store(last_ob, action, reward, ob, done)
        if first_in_seq:
            self.memory.store(*first_in_seq)

    def update(self, global_step: int):
        if len(self.memory) >= self.batch_size:
            self.update_model()

            self.beta = max(
                self.buffer_beta_min,
                self.beta
                - (self.buffer_beta_max - self.buffer_beta_min)
                * self.buffer_beta_decay,
            )
            self._target_update()
            self.dqn.reset_noise()
            self.dqn_target.reset_noise()

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]
        elementwise_loss = self._compute_dqn_loss(samples)

        samples = self.memory_n.sample_batch_from_idxs(indices)
        gamma = self.gamma ** self.n_step
        n_loss = self._compute_dqn_loss(samples, gamma)
        elementwise_loss += n_loss

        loss = torch.mean(elementwise_loss * weights)
        self.loss_per_batch += loss
        self.episode_step += 1
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)

        for p, v in self.dqn.named_parameters():
            if torch.isnan(v.grad).any():
                logging.warn("param {} nan".format(p))

        self.optimizer.step()

        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps

        self.memory.update_priorities(indices, new_priorities.squeeze())

        return loss.item()

    def metrics(self, episode_num: int):
        if episode_num % 10:
            loss_per_batch_mean = (
                (self.loss_per_batch / self.episode_step).detach().numpy()
                if self.episode_step != 0
                else 0
            )
            self.loss_per_batch = 0
            self.episode_step = 0
            grads = {}
            for i, (name, p) in enumerate(self.dqn.named_parameters()):
                if p.grad is not None:
                    avg = torch.mean(p).detach().numpy()
                    std = torch.std(p).detach().numpy() if p.squeeze().dim() != 0 else 0

                    grads.update({name + "_std": std, name + "_mean": avg})
            return {"loss_per_batch": loss_per_batch_mean, **grads}

    def _compute_dqn_loss(
        self, samples: Dict[str, np.ndarray], gamma: float = None
    ) -> torch.Tensor:
        """Return categorical dqn loss."""
        if gamma is None:
            gamma = self.gamma
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                )
                .long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_update(self):
        for w, w_target in zip(self.dqn.parameters(), self.dqn_target.parameters()):
            w_target.data = w_target.data * (1 - self.tau) + w.data * self.tau

    @staticmethod
    def get_default_parameters():
        return {
            "memory_size": 2000,
            "batch_size": 32,
            "seed": 777,
            "gamma": 0.99,
            "tau": 0.01,
            "buffer_alpha": 0.6,
            "buffer_beta_max": 0.9,
            "buffer_beta_min": 0.1,
            "buffer_beta_decay": 1 / 2000,
            "prior_eps": 1e-6,
            "k_mixtures": 5,
            "eps": 0.2,
            "atom_size": 51,
            "n_step": 3,
            "v_min": -600,
            "v_max": 300,
        }
