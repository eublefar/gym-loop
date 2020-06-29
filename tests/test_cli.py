import os
from click.testing import CliRunner
from gym_loop.cli import create_default


def test_create_default():
    runner = CliRunner()
    result = runner.invoke(create_default, ["-c", "my_config.yaml"])
    assert result.exit_code == 0
    assert os.path.isfile("./my_config.yaml")
    os.remove("./my_config.yaml")
