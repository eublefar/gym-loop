from typing import Dict, List, Tuple
from .base_agent import BaseAgent
from .replay_buffers.replay_buffer import ReplayBuffer
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import logging


class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)


class PytorchRainbowDqn(BaseAgent):
    def __init__(self, **params):
        super().__init__(**params)

        obs_dim = self.observation_space.shape[0]
        action_dim = self.action_space.n

        self.memory = ReplayBuffer(obs_dim, self.memory_size, self.batch_size)
        self.epsilon = self.max_epsilon

        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim).to(self.device).train()
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
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
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = self.action_space.sample()
        else:
            selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        return selected_action

    def memorize(
        self,
        last_ob: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        ob: np.ndarray,
        global_step: int,
    ):
        return self.memory.store(last_ob, action, reward, ob, done)

    def update(self, global_step: int):
        if len(self.memory) >= self.batch_size:
            self.update_model()
            self.epsilon = max(
                self.min_epsilon,
                self.epsilon
                - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay,
            )

            self._target_update()

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)
        self.loss_per_batch += loss
        self.episode_step += 1
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def metrics(self, episode_num: int):
        if episode_num % 10:
            loss_per_batch_mean = (
                (self.loss_per_batch / self.episode_step)
                if self.episode_step != 0
                else 0
            )
            self.loss_per_batch = 0
            self.episode_step = 0
            grads = {}
            for i, p in enumerate(self.dqn.parameters()):
                if p.grad is not None:
                    name = p.name if p.name else str(i) + "_grad"
                    avg = torch.mean(p.grad)
                    std = torch.std(p.grad)
                    grads.update({name + "_std": std, name + "_mean": avg})
            return {
                "loss_per_batch": loss_per_batch_mean,
                "epsilon": self.epsilon,
                **grads,
            }

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = (
            self.dqn_target(next_state)
            .gather(1, self.dqn(next_state).argmax(dim=1, keepdim=True))
            .detach()
        )
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        # logging.info("target {}".format(target))
        # logging.info("curr_q {}".format(curr_q_value))
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def _target_update(self):
        for w, w_target in zip(self.dqn.parameters(), self.dqn_target.parameters()):
            w_target.data = w_target.data * (1 - self.tau) + w.data * self.tau

    @staticmethod
    def get_default_parameters():
        return {
            "memory_size": 2000,
            "batch_size": 32,
            "epsilon_decay": 1 / 2000,
            "seed": 777,
            "max_epsilon": 1.0,
            "min_epsilon": 0.1,
            "gamma": 0.99,
            "tau": 0.01,
        }
