"""Console script for gym-loop."""
import sys
import click
import yaml
from gym_loop.gym_loop import (
    train_agent,
    eval_agent,
    get_default_params,
    validate_module_str,
)


@click.group()
def main():
    """Console script for gym-loop."""
    return 0


@main.command()
@click.option(
    "-c",
    "--run-config",
    required=True,
    type=click.File("r"),
    help="Path to run parameters YAML file",
)
def train(run_config):
    """Run agent training experiment specified in run config"""
    train_agent(yaml.load(run_config.read()))


@main.command()
@click.option(
    "-c",
    "--run-config",
    required=True,
    type=click.File("r"),
    help="Path to run parameters YAML file",
)
def eval(run_config):
    """Evaluate agent specified in run config without updating"""
    eval_agent(yaml.load(run_config.read()))


def validate_module_option(ctx, param, value):
    if validate_module_str(value):
        return value
    else:
        click.echo("Module string must be of format package.module:Class")
        ctx.exit()


@main.command()
@click.option(
    "-c",
    "--run-config",
    required=True,
    type=click.File("w"),
    help="Path to where yaml parameter file will be generated",
)
@click.option(
    "-a",
    "--agent",
    default="gym_loop.agents.random_agent:RandomAgent",
    type=click.STRING,
    callback=validate_module_option,
    help="Agent string for which spec will be generated",
)
@click.option(
    "-a",
    "--loop",
    default="gym_loop.loops.default_loop:DefaultLoop",
    type=click.STRING,
    callback=validate_module_option,
    help="Loop string for which spec will be generated",
)
def create_default(run_config, agent, loop):
    """Produce default run config for the agent and loop specified"""
    params = get_default_params(agent, loop)
    yaml_str = yaml.dump(params)
    run_config.write(yaml_str)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
