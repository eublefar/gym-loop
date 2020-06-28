#!/usr/bin/env python

"""Tests for `botbowl_bot` package."""

from botbowl_bot import botbowl_bot
from botbowl_bot.loops.default_loop import DefaultLoop


def test_generate_build():
    default_loop = "botbowl_bot.loops.default_loop:DefaultLoop"
    default_agent = "botbowl_bot.agents.random_agent:RandomAgent"

    params = botbowl_bot.get_default_params(default_agent, default_loop)

    botbowl_bot.train_agent(params)

    botbowl_bot.eval_agent(params)


class TestClass:
    @staticmethod
    def check_is_class():
        return "abcdefg"


def test_module_str_to_class():
    cls_type = botbowl_bot.module_str_to_class(__file__ + ":TestClass")
    assert cls_type.check_is_class() == TestClass.check_is_class()

    cls_type = botbowl_bot.module_str_to_class(
        "botbowl_bot.loops.default_loop:DefaultLoop"
    )
    assert cls_type.get_default_parameters() == DefaultLoop.get_default_parameters()


def test_validate_module_str():
    assert botbowl_bot.validate_module_str("abc.abc:Class")
    assert botbowl_bot.validate_module_str(__file__ + ":Class")
    assert not botbowl_bot.validate_module_str("14abc.abc:Class")
    assert not botbowl_bot.validate_module_str("abc.abc:14Class")
    assert not botbowl_bot.validate_module_str("abc.abc:Cl###$ass")
