"""Tests for `random_agent` module."""

import gym

from gym_loop.agents.random_agent import RandomAgent


def test_act():
    mock_env = gym.make("CartPole-v0")
    agent = RandomAgent(action_space=mock_env.action_space)
    action = agent.act(mock_env.observation_space.sample(), 0)
    assert isinstance(action, type(mock_env.action_space.sample()))


def test_memorize():
    pass


def test_update():
    pass
