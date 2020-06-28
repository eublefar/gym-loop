#!/usr/bin/env python

"""Tests for `gym_loop` package."""

from gym_loop import gym_loop
from gym_loop.loops.default_loop import DefaultLoop


def test_generate_build():
    default_loop = "gym_loop.loops.default_loop:DefaultLoop"
    default_agent = "gym_loop.agents.random_agent:RandomAgent"

    params = gym_loop.get_default_params(default_agent, default_loop)

    gym_loop.train_agent(params)

    gym_loop.eval_agent(params)


class TestClass:
    @staticmethod
    def check_is_class():
        return "abcdefg"


def test_module_str_to_class():
    cls_type = gym_loop.module_str_to_class(__file__ + ":TestClass")
    assert cls_type.check_is_class() == TestClass.check_is_class()

    cls_type = gym_loop.module_str_to_class("gym_loop.loops.default_loop:DefaultLoop")
    assert cls_type.get_default_parameters() == DefaultLoop.get_default_parameters()


def test_validate_module_str():
    assert gym_loop.validate_module_str("abc.abc:Class")
    assert gym_loop.validate_module_str(__file__ + ":Class")
    assert not gym_loop.validate_module_str("14abc.abc:Class")
    assert not gym_loop.validate_module_str("abc.abc:14Class")
    assert not gym_loop.validate_module_str("abc.abc:Cl###$ass")
