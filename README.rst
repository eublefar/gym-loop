===========
Gym Loop
===========


.. image:: https://img.shields.io/pypi/v/gym-loop.svg
        :target: https://pypi.python.org/pypi/gym-loop

.. image:: https://img.shields.io/travis/eublefar/gym-loop.svg
        :target: https://travis-ci.com/eublefar/gym-loop

.. image:: https://readthedocs.org/projects/botbowl-bot/badge/?version=latest
        :target: https://botbowl-bot.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/eublefar/gym-loop/shield.svg
     :target: https://pyup.io/repos/github/eublefar/gym-loop/
     :alt: Updates



Ever got tired from copying same agent training loop for each of your RL gym agents? 
Gym Loop got you covered!
Minimalistic framework for running experiments on Openai Gym environments. 


* Free software: MIT license
* Documentation: https://gym-loop.readthedocs.io.

Features
--------

* Deep learning framework agnostic
* Easy to track experiments with simple YAML configs
* Easy agent packaging
* Super simple (<300 lines of code) 

Quickstart
-------

This tool revolves around YAML configs, so lets look at one.::

        gym-loop create-default --run-config default-run.yaml

It will create default-run.yaml that will look like this::

        agent:
                class: gym_loop.agents.random_agent:RandomAgent
                parameters: {}
        env:
                imports: []
                name: Pendulum-v0
                parameters: {}
        loop:
                class: gym_loop.loops.default_loop:DefaultLoop
                parameters:
                        episodes_per_checkpoint: 10
                        eval_episodes: 10
                        eval_logdir: ./eval_logs
                        eval_record: false
                        eval_record_dir: ./eval_records
                        eval_render: false
                        logdir: ./logs
                        max_episode_len: 5000
                        max_episodes: 5
                        random_episodes: 0
                        record: false
                        record_dir: ./episode_records
                        render: false
                        time_per_checkpoint: null

As you can see this run spec will create RandomAgent on pendulum env with default loop.
If you would like to test non-standard gym environment you can add it's package into imports list.
Lets change render to true and run it with::

        gym-loop train -c default-run.yaml

You should see random agent in action.

To create an agent gym_loop.agents.BaseAgent class should be implemented::

        class BaseAgent:
                @staticmethod
                def get_default_parameters():
                        """Specifies tweakable parameters for agents
                        
                        Returns:
                                dict: default parameters for the agent
                        """
                        raise NotImplementedError()

                def act(self, state, episode_num):
                        """Retrieves agent's action upon state"""
                        raise NotImplementedError()

                def memorize(self, last_ob, action, reward, done, ob):
                        """Called after environment steps on action, arguments are classic SARSA tuple"""
                        raise NotImplementedError()

                def update(self, episode_num):
                        """Called immediately after memorize"""
                        raise NotImplementedError()

                def __init__(self, **params):
                        super().__init__()
                        self.parameters = self.get_default_parameters()
                        self.parameters.update(params)

Static method get_default_parameters returns parameters that are added to yaml config.
Internally class consumes it through self.parameters member.

Then just generate default run config using gym-loop util. 
Agent can be refered to with both "filepath.py:Classname" and "package.module:Classname" format (package must be installed in the environment). e.g.::

        gym-loop create-default --agent "~/my_agent.py:MyAgent" --run-config my-agent-default-run.yaml

This will output my-agent-default-run.yaml file.
This run config can then be configured and used with::

      gym-loop train -c my-agent-default-run.yaml

      # or to run environment without memorize and update
      
      gym-loop evaluate -c my-agent-default-run.yaml

Different loop logic can also be implemented through gym_loop.loops.BaseLoop

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
