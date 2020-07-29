from typing import Dict, List, Tuple, Deque
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
from collections import deque

logging.basicConfig(level=logging.INFO)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        feature = self.feature_layer(x)
        policy_hid = F.relu(self.policy_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        actions = F.softmax(self.policy_layer(feature), dim=-1)
        value = self.value_layer(feature)

        return actions, value

    def reset_noise(self):
        for module in self.modules():
            if hasattr(module, "reset_noise") and not module is self:
                module.reset_noise()


class PPO(BaseAgent):
    def __init__(self, **params: Dict):
        super().__init__(**params)

        obs_dim = self.observation_space.shape[0]
        action_dim = self.action_space.n
        self.beta = self.buffer_beta_min
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.memory = PrioritizedReplayBuffer(
        #     obs_dim=obs_dim,
        #     size=self.n_steps * self.n_envs,
        #     batch_size=self.batch_size,
        #     alpha=self.buffer_alpha,
        #     gamma=self.gamma,
        # )
        self.memory = ReplayBuffer(
            obs_dim=obs_dim,
            size=self.n_steps * self.n_envs * 2,
            batch_size=self.batch_size,
        )
        self.model = NoisyActorCritic(obs_dim, action_dim).to(self.device).train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.buffers = [deque(maxlen=self.n_steps) for _ in range(self.n_envs)]

        self.last_values = [None] * self.n_envs
        self.last_action_dist = [None] * self.n_envs

        self.uploaded = [False for i in range(self.n_envs)]

        self.policy_loss_sum = 0
        self.policy_loss_num = 0
        self.value_loss_sum = 0
        self.value_loss_num = 0
        self.advantages_sum = 0
        self.advantages_num = 0
        self.values_sum = 0
        self.values_num = 0

    def act(self, state: np.ndarray, episode_num: int, env_id: int = 0):
        """Retrieves agent's action upon state"""
        action_distr, value = self.model(torch.FloatTensor(state).to(self.device))
        action = Categorical(action_distr).sample()
        self.last_values[env_id] = value.detach().numpy()
        self.last_action_dist[env_id] = action_distr.detach().data.numpy()
        return action.detach().cpu().numpy()

    def memorize(
        self,
        last_ob: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        ob: np.ndarray,
        global_step: int,
        env_id: int = 0,
    ):
        """Called after environment steps on action, arguments are classic SARSA tuple"""

        if self.last_values[env_id] is None:
            raise RuntimeError("No value stored from previous action")
        last_value = self.last_values[env_id]
        self.last_values[env_id] = None

        if self.last_action_dist[env_id] is None:
            raise RuntimeError("No action distribution stored from previous action")
        last_action_dist = self.last_action_dist[env_id]
        self.last_action_dist[env_id] = None

        buffer = self.buffers[env_id]
        transition = (last_ob, action, reward, ob, done, last_value, last_action_dist)
        buffer.append(transition)
        if len(buffer) and (len(buffer) % self.n_steps == 0) or done:
            self.uploaded[env_id] = True
            advantages, values = self._gae(np.asarray(buffer), self.lam)
            for i, transition in enumerate(buffer):
                try:
                    self.memory.store(
                        *transition[:-2],
                        data=dict(
                            adv=advantages[i],
                            value=values[i],
                            action_dist=transition[-1],
                        )
                    )
                except IndexError:
                    print(advantages)
                    print(values)
                    print(transition)
            buffer.clear()

    def _gae(
        self, transitions: Deque[Tuple], lam: float = 1.0
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """computes advantages and value targets"""
        rewards = torch.FloatTensor(transitions[:, 2].astype(np.float))
        values = torch.FloatTensor(np.stack(transitions[:, 5]))

        if transitions[-1, 4]:
            last_v = 0
        else:
            # print("not done", len(transitions))
            last_state = torch.FloatTensor(transitions[-1, 3].astype(np.float))
            _, last_v = self.model(last_state)
            last_v = last_v.detach()

        if lam == 1:
            gamma_pow = (
                torch.arange(start=0, end=rewards.shape[0])
                .unsqueeze(0)
                .expand(rewards.shape[0], -1)
            ).data.numpy()
            np_pows = np.stack(
                [np.roll(gamma_pow[i, :], shift=i) for i in range(rewards.shape[0])]
            )
            gamma_pows = torch.triu(torch.FloatTensor(np_pows))
            gamma_tri = torch.triu(torch.pow(self.gamma, gamma_pows))
            returns = torch.matmul(gamma_tri, rewards.unsqueeze(-1))

            emp_values = returns + last_v * (
                self.gamma ** (gamma_pows[:, -1] + 1).unsqueeze(-1)
            )
            advantages = emp_values - values

            advantages = advantages.squeeze(-1)
        else:
            gae = 0
            emp_values = []
            for value, reward in list(zip(values, rewards))[::-1]:
                gae = gae * self.gamma * lam
                gae = gae + reward + self.gamma * last_v - value
                last_v = value
                emp_values.append(gae + value)
            emp_values = emp_values[::-1]
            emp_values = torch.cat(emp_values).detach()
        advantages = emp_values - values.squeeze(-1)
        return advantages, emp_values

    def update(self, episode_num: int):
        """Called immediately after memorize"""
        if all(self.uploaded):
            self.uploaded = [False for i in range(self.n_envs)]
            for k in range(self.epochs):
                for samples in self.memory.sample_iterator():

                    elem_loss = self._compute_loss(samples)
                    loss = torch.mean(elem_loss)
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()
                    self.model.reset_noise()
            self.memory.empty()
            self.beta = max(
                self.buffer_beta_min,
                self.beta
                - (self.buffer_beta_max - self.buffer_beta_min)
                * self.buffer_beta_decay,
            )

    def _compute_loss(self, samples: Dict) -> torch.Tensor:
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        values = torch.FloatTensor(
            np.stack([record["value"] for record in samples["data"]])
        ).to(device)

        advantages = torch.FloatTensor(
            np.stack([record["adv"] for record in samples["data"]])
        ).to(device)

        self.advantages_num += 1
        self.advantages_sum += advantages.mean()
        advantages = (advantages - advantages.mean()) / advantages.std()

        action_dist = torch.FloatTensor(
            np.stack([record["action_dist"] for record in samples["data"]])
        ).to(device)

        action_dist_new, value_pred = self.model(state)

        value_loss = F.smooth_l1_loss(value_pred.squeeze(-1), values, reduction="none")

        action_dist = Categorical(action_dist)
        action_dist_new = Categorical(action_dist_new)

        old_logprobs = action_dist.log_prob(action)
        new_logprobs = action_dist_new.log_prob(action)

        ratios = torch.exp(new_logprobs - old_logprobs)
        policy_gain = ratios * advantages
        policy_gain_clipped = (
            torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        )
        policy_loss = -torch.min(policy_gain, policy_gain_clipped)

        self.policy_loss_num += 1
        self.policy_loss_sum += policy_loss.mean().detach()
        self.value_loss_num += 1
        self.value_loss_sum += value_loss.mean().detach()
        self.values_num += 1
        self.values_sum += value_pred.mean().detach()
        return (
            policy_loss
            + self.value_loss_coef * value_loss
            - self.entropy_reg_coef * action_dist_new.entropy()
        )

    def metrics(self, episode_num: int) -> Dict:
        """Returns dict with metrics to log in tensorboard"""
        m = (
            {
                # "max_priority": self.memory.max_priority,
                "value_loss": self.value_loss_sum / self.value_loss_num,
                "value": self.values_sum / self.values_num,
                "advantage": self.advantages_sum / self.advantages_num,
                "policy_loss": self.policy_loss_sum / self.policy_loss_num,
            }
            if self.policy_loss_num
            # else {"max_priority": self.memory.max_priority}
            else {}
        )
        self.policy_loss_sum = 0
        self.policy_loss_num = 0
        self.value_loss_sum = 0
        self.value_loss_num = 0
        self.advantages_sum = 0
        self.advantages_num = 0
        self.values_sum = 0
        self.values_num = 0
        return m

    @staticmethod
    def get_default_parameters() -> Dict:
        """Specifies tweakable parameters for agents
        
        Returns:
            dict: default parameters for the agent
        """
        return {
            "lam": 0.95,
            "gamma": 0.99,
            "epochs": 10,
            "entropy_reg_coef": 0.01,
            "value_loss_coef": 0.5,
            "eps_clip": 0.2,
            "lr": 1e-4,
            "use_per": False,
            "n_envs": 1,
            "n_steps": 25,
            "batch_size": 32,
            "prior_eps": 1e-8,
            "buffer_alpha": 0.6,
            "buffer_beta_max": 0.9,
            "buffer_beta_min": 0.1,
            "buffer_beta_decay": 1 / 2000,
        }

