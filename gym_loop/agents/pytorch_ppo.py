from typing import Dict, List, Tuple, Deque, Any
from .base_agent import BaseAgent
from .replay_buffers.replay_buffer import ReplayBuffer
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import logging
from torch.distributions.categorical import Categorical
from contextlib import suppress
from collections import deque
from math import exp

try:
    from torch.cuda.amp import autocast, GradScaler

    MIXED_PREC = True
except ImportError:
    print("Mixed precision training is not available")
    MIXED_PREC = False
logging.basicConfig(level=logging.INFO)


class PPO(BaseAgent):
    def __init__(self, **params: Dict):
        super().__init__(**params)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = ReplayBuffer(
            size=self.n_steps * self.n_envs * 2, batch_size=self.batch_size
        )
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.lr, eps=1e-8
        )
        self.buffers = [deque(maxlen=self.n_steps) for _ in range(self.n_envs)]

        if MIXED_PREC:
            self.scaler = GradScaler()

        self.last_values = [None] * self.n_envs
        self.last_action_logprobs = [None] * self.n_envs
        self.last_actions_probs = [None] * self.n_envs

        self.last_values_batch = None
        self.last_actions_logprobs = None
        self.adv_mean = None
        self.adv_std = None
        self.momentum = 0.25
        self.metrics_dict = {}

        self.uploaded = [False for i in range(self.n_envs)]

        self.grad_norm_sum = 0
        self.grad_norm_num = 0
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
        self.last_values[env_id] = value.cpu().pin_memory()
        self.last_action_logprobs[env_id] = (
            action_distr.log_prob(action)
            .detach()
            .squeeze()
            .to("cpu", non_blocking=True)
        )
        return action.detach()

    def batch_act(self, state_batch, mask, log_prob_override=None):
        outp = self.policy(state_batch)
        action_distrs, values = outp["action_distribution"], outp["values"]
        actions = action_distrs.sample()
        self.last_values_batch = values.detach().cpu().pin_memory()

        self.last_actions_logprobs = (
            action_distrs.log_prob(actions).detach().squeeze().cpu().pin_memory()
            if log_prob_override is None
            else log_prob_override
        )
        self.last_actions_probs = action_distrs.probs.cpu()
        return actions.detach()

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
        
        if self.last_actions_probs is None:
            raise RuntimeError("No action distribution stored from previous action")
        last_actions_probs = self.last_actions_probs
        self.last_actions_probs = None
        
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
                last_actions_probs[env_id, :]
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
                *transition[:-3],
                data=dict(
                    adv=advantages[i],
                    value=values[i],
                    action_logprob=transition[-2],
                    action_prob=transition[-1]
                )
            )
        buffer.clear()

    def _gae(
        self, transitions: Deque[Tuple], lam: float = 1.0
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """computes advantages and value targets"""
        # (batch_size)
        rewards = torch.FloatTensor(transitions[:, 2].astype(np.float))
        if torch.isnan(rewards).any():
            print("new_logprobs")
        # (batch_size, 1)
        values = torch.stack(transitions[:, 5].tolist())
#         if transitions[-1, 4]:
        last_v = 0
#         else:
#             last_state = torch.FloatTensor(transitions[-1, 3].astype(np.float))
#             outp = self.policy(last_state)
#             last_v = outp["values"]
#             last_v = last_v.detach().squeeze()

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
            kl_approx = 0
            self.uploaded = [False for i in range(self.n_envs)]
            for k in range(self.epochs):
                for epoch_step, samples in enumerate(self.memory.sample_iterator()):
                    print(epoch_step)
                    iters = len(samples) // self.sub_batch_size + int(
                        (len(samples) % self.sub_batch_size) != 0
                    )
                    if kl_approx == 0:
                        pass
                    elif (kl_approx/iters) >= (1.5 * self.target_kl):
                        self.beta *= 2
                    elif (kl_approx/iters) <= (self.target_kl/1.5):
                        self.beta /= 2
                    kl_approx = 0
                    self.optimizer.zero_grad()
                    for i in range(iters):
                        print(i)
                        upper = (i + 1) * self.sub_batch_size
                        lower = i * self.sub_batch_size
                        try:
                            elem_loss, kl = self._compute_loss(
                                {
                                    sample_key: sample[lower:upper]
                                    for sample_key, sample in samples.items()
                                }
                            )
                            # TODO: Update beta to self.target_kl
                            kl_approx += kl
                            loss = torch.mean(elem_loss) / iters
                            if MIXED_PREC:
                                self.scaler.scale(loss).backward()
                            else:
                                loss.backward()
                            del loss, elem_loss
                        except RuntimeError as e:
                            print(e)
                            print(
                                "lower %d, upper %d, samples length %d, iters %d"%(
                                lower, upper, len(samples), iters)
                            )

                    if MIXED_PREC:
                        self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.policy.model.parameters(), self.gradient_clip_norm)
                    clip_grad_norm_(self.policy.value_head.parameters(), self.gradient_clip_norm)

#                     for p in self.policy.parameters():
#                         p.grad.data.clamp_(min=-0.5, max=0.5)
                    
                    total_norm = 0
                    for p in self.policy.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2).cpu()
                            total_norm += param_norm.item() ** 2
                        else:
                            print("param grad is none", p)
                    total_norm = total_norm ** (1.0 / 2)
                    if total_norm == total_norm:
                        self.update_means({"grad_norm": total_norm})
                    else:
                        print("encountered nan")
                    self.metrics_dict["beta"] = self.beta
                        
                    if (kl_approx/iters) >= 0.4:
                        print("ignoring batch", kl_approx/iters)
                        self.set_lr(0)
                    if MIXED_PREC:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.policy.reset_noise()
                    self.set_lr(self.lr)
                    
                        
            self.memory.empty()
    
    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            
    def _compute_loss(self, samples: Dict) -> torch.Tensor:

        state = samples["obs"]
        action = torch.stack(samples["acts"].tolist()).to(
            self.device, non_blocking=True
        )
        values = torch.stack([record["value"] for record in samples["data"]]).to(
            self.device, non_blocking=True
        )
        advantages = (
            torch.stack([record["adv"] for record in samples["data"]])
            .squeeze(-1)
            .to(self.device, non_blocking=True)
        )
        
        probs = (
            torch.stack([record["action_prob"] for record in samples["data"]])
            .squeeze(-1)
            .to(self.device, non_blocking=True)
        ).detach()

        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_mean == adv_mean:
            self.adv_mean = (
                self.adv_mean * (1 - self.momentum) + adv_mean * (self.momentum)
                if self.adv_mean is not None else
                adv_mean
            )
        if adv_std == adv_std:
            self.adv_std = (
                self.adv_std * (1 - self.momentum) + adv_std * (self.momentum)
                if self.adv_std is not None else
                adv_std
            )
        advantages = ((advantages - self.adv_mean) / self.adv_std) if self.adv_std != 0 else 0

        old_logprobs = torch.stack(
            [record["action_logprob"] for record in samples["data"]],
        ).to(self.device, non_blocking=True)
        outp = self.policy(state)
        action_dist_new, value_pred = outp["action_distribution"], outp["values"]
#         with autocast() if MIXED_PREC else suppress():
        if torch.isnan(value_pred).any():
            print("value_pred")
        if torch.isnan(values).any():
            print("values")
        value_loss = F.smooth_l1_loss(value_pred , values , reduction="none")

        new_logprobs = action_dist_new.log_prob(action)
        ratios = torch.exp(new_logprobs  - old_logprobs )

        kl_elem = (action_dist_new.probs * (torch.log(action_dist_new.probs) - torch.log(probs))).sum(dim=-1)
        kl_approx = kl_elem.mean().item()

        if (kl_elem < 0).any():
            print("KL Cant be < 0 !!")
            print(kl_elem)
            print(new_logprobs)
            print(old_logprobs)
            raise ValueError()
#             ratios_clipped = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
#             policy_gain = ratios * advantages 
#             policy_gain_clipped = ratios_clipped * advantages 
#             policy_loss = -torch.where(
#                 torch.abs(policy_gain) < torch.abs(policy_gain_clipped), 
#                 policy_gain, 
#                 policy_gain_clipped
#             )
        policy_gain = ratios * advantages 
        policy_loss = -policy_gain
        
        if any(policy_loss >= 100):
            print("Very big loss")
#             print(policy_loss)
            print(policy_gain)#, policy_gain_clipped)
        if torch.isnan(new_logprobs).any():
            print("new_logprobs")
        if torch.isnan(advantages).any():
            print("advantages")
#         if torch.isnan(policy_gain_clipped).any():
#             print("policy_gain_clipped")
        if torch.isnan(policy_loss).any():
            print("policy_loss")
        self.update_means(
            {
                "advantages": advantages.cpu().mean().detach().numpy(),
                "policy_loss": policy_loss.cpu().mean().detach().numpy(),
                "value_loss": value_loss.cpu().mean().detach().numpy(),
                "values": value_pred.cpu().mean().detach().numpy(),
                "entropy": action_dist_new.entropy().cpu().mean().detach().numpy(),
            }
        )
        return (
            policy_loss
            + self.value_loss_coef * value_loss.squeeze()
            - self.entropy_reg_coef * action_dist_new.entropy().squeeze()
            + self.beta * kl_elem.mean()
        ), kl_approx

    def update_means(self, values: Dict[str, float]):
        for k, v in values.items():
            setattr(self, k + "_num", getattr(self, k + "_num") + 1)
            setattr(self, k + "_sum", getattr(self, k + "_sum") + v)

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
                "grad_norm": self.grad_norm_sum / self.grad_norm_num
                if self.grad_norm_num != 0
                else 0,
            }
            m["perplexity"] = exp(m["entropy"])
            m.update(self.metrics_dict)
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
            self.grad_norm_sum = 0
            self.grad_norm_num = 0
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
            "beta": 100,
            "target_kl": 0.015,
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
            "gradient_clip_norm": 0.5,
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

