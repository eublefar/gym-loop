"""Thanks to https://github.com/Curt-Park/rainbow-is-all-you-need"""
from typing import Dict, List, Tuple
from .base_agent import BaseAgent
from .replay_buffers.per_buffer import PrioritizedReplayBuffer
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
    def __init__(self, in_dim: int, out_dim: int, k_mixtures: int = 5):
        """Initialization."""
        super(GMNetwork, self).__init__()

        param_num = k_mixtures * 3  # weight, mu, sigma
        self.k = k_mixtures
        self.param_num = param_num
        self.out_dim = out_dim

        self.layers = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU())
        # set advantage layer
        self.advantage_layer = nn.Sequential(
            NoisyLinear(128, 128), nn.ReLU(), NoisyLinear(128, out_dim * param_num)
        )
        # # set value layer
        # self.value_layer = nn.Sequential(
        #     NoisyLinear(128, 128), nn.ReLU(), NoisyLinear(128, param_num)
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        features = self.layers(x)
        advantage = self.advantage_layer(features)
        advantage = advantage.reshape((-1, self.out_dim, self.k, 3))
        # value = self.value_layer(features)
        # value = value.reshape((-1, 1, self.k, 3))

        advantage_pi, advantage_mu, advantage_sig = torch.unbind(advantage, dim=-1)
        # value_pi, value_mu, value_sig = torch.unbind(value, dim=-1)

        # (batch, 1, k)
        # mean_advantage_mu = advantage_mu.mean(dim=1, keepdim=True)

        # (batch, 1, k)
        # value_pi = value_pi
        # print(advantage_pi.shape)
        final_pi = torch.nn.functional.softmax(advantage_pi, dim=-1)
        # final_pi = torch.nn.functional.softmax(value_pi * advantage_pi, dim=-1)

        advantage_mu = advantage_mu
        # final_mu = value_mu + advantage_mu
        final_mu = advantage_mu

        # value_sig = torch.nn.functional.relu(value_sig)
        advantage_sig = torch.nn.functional.relu(advantage_sig)

        # final_sig = value_sig + advantage_sig
        final_sig = advantage_sig
        q_gm = torch.stack([final_pi, final_mu, final_sig], dim=-1)

        # (batch, out_dim, k, pi + mu + sigma^2)
        return q_gm

    def sample(self, q_gm):

        # 3 x (batch, out_dim, k)
        pi, mu, sigma_sq = torch.unbind(q_gm, dim=-1)

        # (batch, out_dim, 1)
        mixture_choice = Categorical(probs=pi).sample().unsqueeze(-1)

        # (batch, out_dim, 1)
        mu_choice = mu.gather(-1, mixture_choice)
        mu_out = mu_choice.squeeze()

        # (batch, out_dim, 1)
        sigma_sq_choice = sigma_sq.gather(-1, mixture_choice)
        sigma_sq_out = sigma_sq_choice.squeeze()

        # (batch, out_dim)
        return Normal(loc=mu_out, scale=torch.sqrt(sigma_sq_out)).sample()

    def reset_noise(self):
        # for module in self.value_layer.modules():
        #     if hasattr(module, "reset_noise"):
        #         module.reset_noise()
        for module in self.advantage_layer.modules():
            if hasattr(module, "reset_noise"):
                module.reset_noise()


class PytorchRainbowDqn(BaseAgent):
    def __init__(self, **params):
        super().__init__(**params)

        obs_dim = self.observation_space.shape[0]
        action_dim = self.action_space.n

        self.memory = PrioritizedReplayBuffer(
            obs_dim, self.memory_size, self.batch_size, alpha=self.buffer_alpha
        )
        self.beta = self.buffer_beta_min
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # networks: dqn, dqn_target
        self.dqn = (
            GMNetwork(obs_dim, action_dim, k_mixtures=self.k_mixtures)
            .to(self.device)
            .train()
        )
        self.dqn_target = GMNetwork(obs_dim, action_dim, k_mixtures=self.k_mixtures).to(
            self.device
        )
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
        # if torch.rand([]) >= self.eps:
        q_pred_gaussian_mixture = self.dqn(torch.FloatTensor(state).to(self.device))
        q_values = self.dqn.sample(q_pred_gaussian_mixture)
        selected_action = q_values.argmax().detach().cpu().numpy()
        # else:
        # selected_action = self.action_space.sample()
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
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        elementwise_loss = self._compute_dqn_loss(samples)
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

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Z(state, action)
        # (batch, 1, k, 3)
        curr_q_gm = self.dqn(state)
        curr_q_gm = curr_q_gm.gather(
            1, action.view([32, 1, 1, 1]).repeat([1, 1, self.k_mixtures, 3])
        )

        # TZ(state, action)

        # 3 x (batch, action, k)
        next_pi, next_mu, next_sigma_sq = torch.unbind(
            self.dqn_target(next_state).detach(), dim=-1
        )
        # (batch, actions)
        next_expected = (next_pi * next_mu).sum(dim=-1)
        # (batch, 1, 1)
        next_actions = next_expected.argmax(dim=-1, keepdim=True).unsqueeze(-1)

        # 3 x (batch, action, k)
        # next_pi, next_mu, next_sigma_sq = torch.unbind(
        #     self.dqn_target(next_state), dim=-1
        # )
        # (batch, 1, k)
        next_pi = next_pi.gather(1, next_actions.repeat([1, 1, self.k_mixtures]))
        # (batch, 1, k)
        next_mu = next_mu.gather(1, next_actions.repeat([1, 1, self.k_mixtures]))
        # (batch, 1, k)
        next_sigma_sq = next_sigma_sq.gather(
            1, next_actions.repeat([1, 1, self.k_mixtures])
        )

        target_gm_pi = next_pi
        mask = 1 - done

        target_gm_mu = reward.unsqueeze(-1) + self.gamma * next_mu * mask.unsqueeze(-1)
        target_gm_sigma_sq = self.gamma * self.gamma * next_sigma_sq

        target_gm = torch.stack(
            [target_gm_pi, target_gm_mu, target_gm_sigma_sq], dim=-1
        ).detach()

        loss = self.jtd_loss(curr_q_gm, target_gm, reduction="none")

        return loss

    def jtd_loss(self, pred_gm, target_gm, reduction="mean"):

        pred_gm_pi, pred_gm_mu, pred_sigma_sq = torch.unbind(pred_gm, dim=-1)
        target_gm_pi, target_gm_mu, target_sigma_sq = torch.unbind(target_gm, dim=-1)

        aa = self.calculate_integral(
            pred_gm_pi, pred_gm_mu, pred_sigma_sq, pred_gm_pi, pred_gm_mu, pred_sigma_sq
        )

        bb = self.calculate_integral(
            target_gm_pi,
            target_gm_mu,
            target_sigma_sq,
            target_gm_pi,
            target_gm_mu,
            target_sigma_sq,
        )
        ab = self.calculate_integral(
            pred_gm_pi,
            pred_gm_mu,
            pred_sigma_sq,
            target_gm_pi,
            target_gm_mu,
            target_sigma_sq,
        )

        loss = aa + bb - 2 * ab

        if (loss < 0).any():
            logging.info("aa" + str(aa))
            logging.info("bb" + str(bb))
            logging.info("ab" + str(ab))
            logging.info("loss" + str(loss))
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise ValueError("Unknown reduction {}".format(reduction))

    def calculate_integral(
        self, gm_pi, gm_mu, gm_sigma_sq, gm_pi_other, gm_mu_other, gm_sigma_sq_other
    ):

        # (batch, actions, k1, k2)
        weights = torch.einsum(
            "baij,bajk->baik", (gm_pi_other.unsqueeze(-1), gm_pi.unsqueeze(-2))
        )
        # (batch, actions, k1, k2)
        sigma_sum = gm_sigma_sq_other.unsqueeze(-1) + gm_sigma_sq.unsqueeze(-2)
        sigma_sum += self.prior_eps

        out_dist = Normal(loc=gm_mu_other.unsqueeze(-1), scale=torch.sqrt(sigma_sum))

        probs = torch.exp(out_dist.log_prob(gm_mu.unsqueeze(-2)))
        if torch.isnan(probs).any():
            logging.info("Got neg probs")
        if torch.isnan(weights).any():
            logging.info("Got neg or zero weights")
        return (probs * weights).sum(dim=-1).sum(dim=-1)

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
        }
