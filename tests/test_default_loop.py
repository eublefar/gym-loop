#!/usr/bin/env python

"""Tests for `deafult_loop` module."""

import gym

from botbowl_bot.loops.default_loop import DefaultLoop
from botbowl_bot.agents.random_agent import RandomAgent


def test_train_run(mocker):
    mock_env = gym.make("CartPole-v0")
    mock_agent = RandomAgent(action_space=mock_env.action_space)
    reset = mocker.patch.object(
        mock_env,
        "reset",
        autospec=True,
        return_value=mock_env.observation_space.sample(),
    )
    step = mocker.patch.object(
        mock_env,
        "step",
        autospec=True,
        return_value=mock_env.observation_space.sample(),
    )
    act = mocker.patch.object(
        mock_agent, "act", autospec=True, return_value=mock_env.action_space.sample(),
    )
    memorize = mocker.patch.object(mock_agent, "memorize", autospec=True)
    update = mocker.patch.object(mock_agent, "update", autospec=True)
    loop = DefaultLoop(
        agent=mock_agent, env=mock_env, max_episodes=1, max_episode_len=1
    )

    loop.train()

    step.assert_called_once()
    reset.assert_called_once()
    act.assert_called_once()
    memorize.assert_called_once()
    update.assert_called_once()


def test_evaluate_run(mocker):
    mock_env = gym.make("CartPole-v0")
    mock_agent = RandomAgent(action_space=mock_env.action_space)
    reset = mocker.patch.object(
        mock_env,
        "reset",
        autospec=True,
        return_value=mock_env.observation_space.sample(),
    )
    step = mocker.patch.object(
        mock_env,
        "step",
        autospec=True,
        return_value=mock_env.observation_space.sample(),
    )
    act = mocker.patch.object(
        mock_agent, "act", autospec=True, return_value=mock_env.action_space.sample(),
    )
    memorize = mocker.patch.object(mock_agent, "memorize", autospec=True)
    update = mocker.patch.object(mock_agent, "update", autospec=True)
    loop = DefaultLoop(
        agent=mock_agent,
        env=mock_env,
        max_episodes=1,
        max_episode_len=1,
        eval_episodes=1,
        eval_render=False,
        eval_record=False,
    )

    loop.evaluate()

    step.assert_called_once()
    reset.assert_called_once()
    act.assert_called_once()
    memorize.assert_not_called()
    update.assert_not_called()
