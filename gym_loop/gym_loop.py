import gym
import importlib
import os
import sys
import re

"""Main module."""


def train_agent(run_params):
    """Run training loop for run parameters
    
    Args:
        run_params(dict) Dictionary from parsed configuration file
    """
    env = build_env(run_params)
    agent = build_agent(run_params, env)
    loop = build_loop(run_params, agent, env)

    loop.train()


def eval_agent(run_params):
    """Run loop without updating agent 

    Args:
        run_params(dict) Dictionary from parsed configuration file
    """
    env = build_env(run_params)
    agent = build_agent(run_params, env)
    loop = build_loop(run_params, agent, env)

    loop.evaluate()


def build_env(params):
    """Create gym env from run parameters"""
    env_params = params["env"]["parameters"]
    env_imports = params["env"]["imports"]
    env_string = params["env"]["name"]
    seed = env_params.pop("seed", None)
    for import_module in env_imports:
        importlib.import_module(import_module)
    env = gym.make(env_string, **env_params)
    if seed is not None:
        env.seed(seed)
    return env


def build_agent(params, env):
    """Create agent from run parameters"""
    agent_params = params["agent"]["parameters"]
    agent_class_string = params["agent"]["class"]
    Agent = module_str_to_class(agent_class_string)
    agent_params["observation_space"] = env.observation_space
    agent_params["action_space"] = env.action_space
    return Agent(**agent_params)


def build_loop(params, agent, env):
    """Create loop from run parameters"""
    loop_params = params["loop"]["parameters"]
    loop_class_string = params["loop"]["class"]
    Loop = module_str_to_class(loop_class_string)
    loop_params["env"] = env
    loop_params["agent"] = agent
    return Loop(**loop_params)


def get_default_params(agent_str, loop_str):
    """Build a dict with default run spec for the agent

    Args:
        loop_str (str): loop module class string of format 'package.module:class' or  'loop_filepath:class'
        agent_str (str): agent module class string of format 'package.module:class' or  'agent_filepath:class'
    
    Returns:
        dict: default run spec dict
    """
    Agent = module_str_to_class(agent_str)
    Loop = module_str_to_class(loop_str)
    return {
        "env": {"name": "Pendulum-v0", "parameters": {}, "imports": []},
        "agent": {"class": agent_str, "parameters": Agent.get_default_parameters()},
        "loop": {"class": loop_str, "parameters": Loop.get_default_parameters()},
    }


def module_str_to_class(module_str):
    """Parse module class string to a class
    
    Args:
        module_str(str) Dictionary from parsed configuration file

    Returns:
        type: class
    """
    if not validate_module_str(module_str):
        raise ValueError("Module string is in wrong format")

    module_path, class_name = module_str.split(":")

    if os.path.isfile(module_path):
        module_name = os.path.basename(module_path).replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)

    return getattr(module, class_name)


def validate_module_str(module_str):
    """Check if string is module class string 
    
    Args:
        module_str(str) A string to test

    Returns:
        bool: Is module class string valid
    """
    module_path, class_name = module_str.split(":")
    identifier = re.compile(r"^[^\d\W]\w*\Z", re.UNICODE)
    classname_correct = re.match(identifier, class_name)
    module_is_path = os.path.isfile(module_path)
    module_is_import_str = all(
        [re.match(identifier, name) for name in module_path.split(".")]
    )

    return classname_correct and (module_is_path or module_is_import_str)
