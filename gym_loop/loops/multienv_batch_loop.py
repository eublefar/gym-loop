from tensorboardX import SummaryWriter
import logging
from gym import wrappers
import numpy as np
from copy import deepcopy

from .base_loop import BaseLoop

logging.basicConfig(level=logging.INFO)


class MultiEnvLoop(BaseLoop):
    @staticmethod
    def get_default_parameters():
        """Get default parameter dictionary for the loop"""
        return {
            "load_from": None,
            "random_episodes": 0,
            "max_episode_len": 5000,
            "record": False,
            "record_dir": "./episode_records",
            "logdir": "./logs",
            "render": False,
            "max_episodes": 50,
            "n_envs": 25,
            "episodes_per_checkpoint": 10,
            "time_per_checkpoint": None,
            "eval_logdir": "./eval_logs",
            "eval_episodes": 10,
            "eval_render": False,
            "eval_record": False,
            "eval_record_dir": "./eval_records",
        }

    def __init__(self, agent, env, **params):
        super().__init__(**params)
        self.writer = SummaryWriter(logdir=self.logdir)
        self.eval_writer = SummaryWriter(logdir=self.logdir)
        self.envs = [deepcopy(env) for _ in range(self.n_envs)]
        self.agent = agent

    def __del__(self):
        self.writer.close()

    def train(self):
        """Training loop"""
        if self.load_from is not None:
            self.policy.load(self.load_from)
        # random loop
        logging.info("Running random episodes {} times".format(self.random_episodes))
        for i in range(self.random_episodes):
            ob = self.env.reset()
            for _ in range(self.max_episode_len):
                ob, reward, done = self._step_random(ob, episode_num=i)
                if done:
                    break

        if self.record:
            logging.info(
                f"Using Gym monitor to save videos, render self.environment flag {self.render}"
            )
            self.envs[0] = wrappers.Monitor(
                self.envs[0], directory=self.record_dir, force=True
            )

        # policy loop
        episode = 0
        done = np.asarray([False for _ in range(self.n_envs)])
        ob = np.stack([env.reset() for env in self.envs])
        reward_per_ep = np.stack([0 for _ in range(self.n_envs)])
        ep_step = np.stack([0 for _ in range(self.n_envs)])

        while episode < self.max_episodes:
            transitions = []
            last_ob = deepcopy(ob)

            actions = self.agent.batch_act(ob, (1 - done))
            for env_id, env in enumerate(self.envs):
                if env_id == 0 and self.render:
                    self.envs[0].render()
                if not done[env_id]:
                    ob[env_id], reward, done[env_id], _ = self.envs[env_id].step(
                        actions[env_id]
                    )
                    reward_per_ep[env_id] += reward
                    ep_step[env_id] += 1
                    transitions.append(
                        (
                            last_ob[env_id],
                            actions[env_id],
                            reward,
                            done[env_id],
                            ob[env_id],
                        )
                    )
                else:
                    transitions.append(None)
            self.agent.batch_memorize(transitions)

            if all(done):
                # print("all_done")
                ob = np.stack([env.reset() for env in self.envs])
                done = np.asarray([False for _ in range(self.n_envs)])
                episode += 1

                metrics = self.agent.metrics(episode)
                if metrics is not None:
                    for name, value in metrics.items():
                        if np.isnan(value):
                            logging.warn("{} has nan".format(name))
                        self.writer.add_scalar(name, value, global_step=episode)

                self.writer.add_scalar(
                    "reward", reward_per_ep.max(), global_step=episode
                )
                self.writer.add_scalar("avg_legth", ep_step.mean(), global_step=episode)
                reward_per_ep = np.stack([0 for _ in range(self.n_envs)])
                ep_step = np.stack([0 for _ in range(self.n_envs)])
                if episode % self.episodes_per_checkpoint == 0 and episode != 0:
                    self.policy.save("checkpoint_{}".format(i))
            self.agent.update(episode)

    def evaluate(self):
        if self.eval_record:
            logging.info(
                f"Using Gym monitor to save videos, render self.environment flag {self.eval_render}"
            )
            self.env = wrappers.Monitor(
                self.env, directory=self.eval_record_dir, force=True
            )
        for i in range(self.eval_episodes):
            ob = self.env.reset()
            reward_per_ep = 0
            for ep_step in range(self.max_episode_len):
                if self.render:
                    self.env.render()
                ob, reward, done = self._step_policy(ob, update=False)
                reward_per_ep += reward
                if done:
                    break
            metrics = self.agent.metrics(i)
            if metrics is not None:
                for name, value in metrics.items():
                    self.eval_writer.add_scalar(name, value, global_step=i)
            self.eval_writer.add_scalar("reward", reward_per_ep, global_step=i)
            self.eval_writer.add_scalar("avg_legth", ep_step, global_step=i)

    def _step_random(self, last_ob, env_id=0):
        action = self.env.action_space.sample()
        ob, reward, done, _ = self.envs[env_id].step(action)
        self.agent.memorize(last_ob, action, reward, done, ob)
        return ob, reward, done

