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
* Easy agent packaging without rewriting argument parsing over and over
* Super simple 

Usage
-------

To create an agent that is compatible with gym loop you have to implement BaseAgent class::

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

When agent is implemented, default run config can then be generated using gym-loop util. 
Agent can be refered to with both "filepath:Classname" and "package.module:Classname" format. e.g.::

        gym-loop create-default --agent "~/my_agent.py:MyAgent" --run-config my-agent-default-run.yaml

This will output my-agent-default-run.yaml file. This run config can then be configured and used with::

      gym-loop train -c my-agent-default-run.yaml

      # or to run environment without memorize and update
      
      gym-loop evaluate -c my-agent-default-run.yaml

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
