from tensorboardX import SummaryWriter
import logging
from gym import wrappers

from .base_loop import BaseLoop


class DefaultLoop(BaseLoop):
    @staticmethod
    def get_default_parameters():
        """Get default parameter dictionary for the loop"""
        return {
            "random_episodes": 0,
            "max_episode_len": 5000,
            "record": False,
            "record_dir": "./episode_records",
            "logdir": "./logs",
            "render": False,
            "max_episodes": 5,
            "episodes_per_checkpoint": 10,
            "time_per_checkpoint": None,
            "eval_logdir": "./eval_logs",
            "eval_episodes": 10,
            "eval_render": False,
            "eval_record": False,
            "eval_record_dir": "./eval_records",
        }

    def __init__(self, agent, env, **params):
        super().__init__()
        self.parameters.update(params)
        print(self.parameters)
        self.writer = SummaryWriter(logdir=self.parameters["logdir"])
        self.eval_writer = SummaryWriter(logdir=self.parameters["logdir"])
        self.env = env
        self.agent = agent

    def __del__(self):
        self.writer.close()

    def train(self):
        """Training loop"""
        # random loop
        logging.info(
            "Running random episodes {} times".format(
                self.parameters["random_episodes"]
            )
        )
        for i in range(self.parameters["random_episodes"]):
            ob = self.env.reset()
            for _ in range(self.parameters["max_episode_len"]):
                ob, reward, done = self._step_random(ob, episode_num=i)
                if done:
                    break

        if self.parameters["record"]:
            logging.info(
                f"Using Gym monitor to save videos, render self.environment flag {self.parameters['render']}"
            )
            self.env = wrappers.Monitor(
                self.env, directory=self.parameters["record_dir"], force=True
            )

        # policy loop
        global_step = 0
        for i in range(
            self.parameters["random_episodes"], self.parameters["max_episodes"]
        ):
            ob = self.env.reset()
            reward_per_ep = 0
            for ep_step in range(self.parameters["max_episode_len"]):
                if self.parameters["render"]:
                    self.env.render()
                global_step += 1
                ob, reward, done = self._step_policy(ob, i)
                reward_per_ep += reward
                if done:
                    break
            self.writer.add_scalar("reward", reward_per_ep, global_step=i)
            self.writer.add_scalar("avg_legth", ep_step, global_step=i)

            if i % self.parameters["episodes_per_checkpoint"] == 0 and i != 0:
                self.agent.save("checkpoint_{}".format(i))

    def evaluate(self):
        if self.parameters["eval_record"]:
            logging.info(
                f"Using Gym monitor to save videos, render self.environment flag {self.parameters['eval_render']}"
            )
            self.env = wrappers.Monitor(
                self.env, directory=self.parameters["eval_record_dir"], force=True
            )
        for i in range(self.parameters["eval_episodes"]):
            ob = self.env.reset()
            reward_per_ep = 0
            for ep_step in range(self.parameters["max_episode_len"]):
                if self.parameters["render"]:
                    self.env.render()
                ob, reward, done = self._step_policy(ob, i, update=False)
                reward_per_ep += reward
                if done:
                    break
            self.eval_writer.add_scalar("reward", reward_per_ep, global_step=i)
            self.eval_writer.add_scalar("avg_legth", ep_step, global_step=i)

    def _step_random(self, last_ob):
        action = self.env.action_space.sample()
        ob, reward, done, _ = self.env.step(action)
        self.agent.memorize(last_ob, action, reward, done, ob)
        return ob, reward, done

    def _step_policy(self, last_ob, episode_num, update=True):
        state = last_ob
        action = self.agent.act(state, episode_num)
        ob, reward, done, _ = self.env.step(action)
        if update:
            self.agent.memorize(last_ob, action, reward, done, ob)
            self.agent.update(episode_num)
        return ob, reward, done
