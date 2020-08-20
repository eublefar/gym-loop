from typing import Dict, List, Tuple, Deque, Any
from .base_agent import BaseAgent
from .replay_buffers.replay_buffer import ReplayBuffer
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import logging
from torch.distributions.categorical import Categorical
from collections import deque

try:
    from torch.cuda.amp import autocast, GradScaler
except ImportError:
    print("Mixed precision training is not available")

logging.basicConfig(level=logging.INFO)


class PPO(BaseAgent):
    def __init__(self, **params: Dict):
        super().__init__(**params)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = ReplayBuffer(
            size=self.n_steps * self.n_envs * 2, batch_size=self.batch_size
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.buffers = [deque(maxlen=self.n_steps) for _ in range(self.n_envs)]

        if torch.cuda.is_available():
            self.scaler = GradScaler()

        self.last_values = [None] * self.n_envs
        self.last_action_logprobs = [None] * self.n_envs

        self.last_values_batch = None
        self.last_actions_logprobs = None

        self.uploaded = [False for i in range(self.n_envs)]

        self.policy_loss_sum = 0
        self.policy_loss_num = 0
        self.value_loss_sum = 0
        self.value_loss_num = 0
        self.advantages_sum = 0
        self.advantages_num = 0
        self.values_sum = 0
        self.values_num = 0
        self.entropy_sum = 0
        self.entropy_num = 0

    def act(self, state: Any, episode_num: int, env_id: int = 0):
        """Retrieves agent's action upon state"""
        outp = self.policy(state)
        action_distr, value = outp["action_distribution"], outp["values"]
        action = action_distr.sample()
        self.last_values[env_id] = value
        self.last_action_logprobs[env_id] = action_distr.log_prob(action).detach().squeeze()
        return action.detach().cpu().numpy()

    def batch_act(self, state_batch, mask):
        outp = self.policy(state_batch)
        action_distrs, values = outp["action_distribution"], outp["values"]
        actions = action_distrs.sample()
        self.last_values_batch = values.detach()
        self.last_actions_logprobs = action_distrs.log_prob(actions).detach().squeeze()
        return actions.detach().cpu().numpy()

    def memorize(
        self,
        last_ob: Any,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        ob: Any,
        global_step: int,
        env_id: int = 0,
    ):
        """Called after environment steps on action, arguments are classic SARSA tuple"""

        if self.last_values[env_id] is None:
            raise RuntimeError("No value stored from previous action")
        last_value = self.last_values[env_id]
        self.last_values[env_id] = None

        if self.last_action_logprobs[env_id] is None:
            raise RuntimeError("No action distribution stored from previous action")
        last_action_logprobs = self.last_action_logprobs[env_id]
        self.last_action_logprobs[env_id] = None
        buffer = self.buffers[env_id]
        transition = (
            last_ob,
            action,
            reward,
            ob,
            done,
            last_value,
            last_action_logprobs,
        )
        buffer.append(transition)
        if len(buffer) and (len(buffer) % self.n_steps == 0) or done:
            self._upload_transitions(env_id)

    def batch_memorize(self, transition_batch):
        if self.last_values_batch is None:
            raise RuntimeError("No value stored from previous action")
        last_values = self.last_values_batch
        self.last_values_batch = None

        if self.last_actions_logprobs is None:
            raise RuntimeError("No action distribution stored from previous action")
        last_actions_logprobs = self.last_actions_logprobs
        self.last_actions_logprobs = None
        for env_id, sards in enumerate(transition_batch):
            if sards is None:
                continue
            buffer = self.buffers[env_id]
            (last_ob, action, reward, done, ob) = sards
            transition = (
                last_ob,
                action,
                reward,
                ob,
                done,
                last_values[env_id],
                last_actions_logprobs[env_id],
            )

            buffer.append(transition)
            if len(buffer) and (len(buffer) % self.n_steps) == 0 or done:
                self._upload_transitions(env_id)

    def _upload_transitions(self, env_id):
        buffer = self.buffers[env_id]
        self.uploaded[env_id] = True
        advantages, values = self._gae(np.asarray(buffer), self.lam)
        for i, transition in enumerate(buffer):
            self.memory.store(
                *transition[:-2],
                data=dict(
                    adv=advantages[i], value=values[i], action_logprob=transition[-1],
                )
            )
        buffer.clear()

    def _gae(
        self, transitions: Deque[Tuple], lam: float = 1.0
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """computes advantages and value targets"""
        # (batch_size)
        rewards = torch.FloatTensor(transitions[:, 2].astype(np.float))
        # (batch_size, 1)
        values = torch.stack(transitions[:, 5].tolist())
        if transitions[-1, 4]:
            last_v = 0
        else:
            last_state = torch.FloatTensor(transitions[-1, 3].astype(np.float))
            outp = self.policy(last_state)
            last_v = outp["values"]
            last_v = last_v.detach().squeeze()

        gae = 0
        emp_values = []
        for value, reward in list(zip(values, rewards))[::-1]:
            gae = gae * self.gamma * lam
            gae = gae + reward + self.gamma * last_v - value
            last_v = value
            emp_values.append(gae + value)
        emp_values = emp_values[::-1]
        emp_values = torch.stack(emp_values).detach()
        # (batch_size)
        advantages = emp_values - values
        return advantages, emp_values

    def update(self, episode_num: int):
        """Called immediately after memorize"""
        if all(self.uploaded):
            print("All uploadedd")
            self.uploaded = [False for i in range(self.n_envs)]
            for k in range(self.epochs):
                for samples in self.memory.sample_iterator():
                    iters = len(samples) // self.sub_batch_size + int(
                        len(samples) % self.sub_batch_size != 0
                    )
                    self.optimizer.zero_grad()
                    for i in range(iters):
                        upper = (i + 1) * self.sub_batch_size
                        lower = i * self.sub_batch_size
                        elem_loss = self._compute_loss(
                            {
                                sample_key: sample[lower:upper]
                                for sample_key, sample in samples.items()
                            }
                        )
                        loss = torch.mean(elem_loss)/iters
                        if torch.cuda.is_available():
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

                    if torch.cuda.is_available():
                        self.scaler.unscale_(self.optimizer)

                    clip_grad_norm_(self.policy.parameters(), 0.5)

                    if torch.cuda.is_available():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.policy.reset_noise()
            self.memory.empty()

    def _compute_loss(self, samples: Dict) -> torch.Tensor:

        state = samples["obs"]
        action = torch.FloatTensor(np.stack(samples["acts"])).to(self.device)
        values = torch.stack([record["value"] for record in samples["data"]]).detach()
        advantages = (
            torch.stack([record["adv"] for record in samples["data"]])
            .detach()
            .squeeze(-1)
        )

        adv_mean = advantages.mean()
        advantages = (advantages - advantages.mean()) / advantages.std()

        old_logprobs = torch.stack(
            [record["action_logprob"] for record in samples["data"]],
        ).detach()
        outp = self.policy(state)
        action_dist_new, value_pred = outp["action_distribution"], outp["values"]
        value_loss = F.smooth_l1_loss(value_pred, values, reduction="none")
        new_logprobs = action_dist_new.log_prob(action)
        ratios = torch.exp(new_logprobs - old_logprobs)
        policy_gain = ratios * advantages
        policy_gain_clipped = (
            torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        )
        policy_loss = -torch.min(policy_gain, policy_gain_clipped)

        self.update_means(
            {
                "advantages": adv_mean,
                "policy_loss": policy_loss.mean(),
                "value_loss": value_loss.mean(),
                "values": value_pred.mean(),
                "entropy": action_dist_new.entropy().mean(),
            }
        )
        return (
            policy_loss
            + self.value_loss_coef * value_loss.squeeze()
            - self.entropy_reg_coef * action_dist_new.entropy().squeeze()
        )

    def update_means(self, values: Dict[str, float]):
        for k, v in values.items():
            setattr(self, k + "_num", getattr(self, k + "_num") + 1)
            setattr(self, k + "_sum", getattr(self, k + "_sum") + v.cpu().detach().numpy())

    def metrics(self, episode_num: int) -> Dict:
        """Returns dict with metrics to log in tensorboard"""
        if self.policy_loss_num:
            m = {
                # "max_priority": self.memory.max_priority,
                "value_loss": self.value_loss_sum / self.value_loss_num,
                "value": self.values_sum / self.values_num,
                "advantage": self.advantages_sum / self.advantages_num,
                "policy_loss": self.policy_loss_sum / self.policy_loss_num,
                "entropy": self.entropy_sum / self.entropy_num,
            }
            self.policy_loss_sum = 0
            self.policy_loss_num = 0
            self.value_loss_sum = 0
            self.value_loss_num = 0
            self.advantages_sum = 0
            self.advantages_num = 0
            self.values_sum = 0
            self.values_num = 0
            self.entropy_sum = 0
            self.entropy_num = 0
        else:
            m = {}
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
            "sub_batch_size": 8,
        }

    @staticmethod
    def get_default_policy() -> Dict:
        """Specifies default policy to use with agent
        
        Returns:
            dict: class string and parameters for the policy
        """
        return {
            "class": "gym_loop.policies.noisy_actor_critic:DiscreteNoisyActorCritic",
            "parameters": {},
        }

