"""Tests for `random_agent` module."""

import gym

from botbowl_bot.agents.random_agent import RandomAgent


def test_act():
    mock_env = gym.make("CartPole-v0")
    print(mock_env.action_space)
    agent = RandomAgent(action_space=mock_env.action_space)
    action = agent.act(mock_env.observation_space.sample(), 0)
    if hasattr(action, "shape"):
        assert action.shape == mock_env.action_space.sample().shape
        assert action.dtype == mock_env.action_space.sample().dtype
    else:
        assert isinstance(action, type(mock_env.action_space.sample()))


def test_memorize():
    pass


def test_update():
    pass
