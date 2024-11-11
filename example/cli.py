"""
Execute lightning cli.

References
----------
- https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html

"""

from lightning.pytorch.cli import LightningCLI


def main() -> None:
    """Execute lightning cli."""
    LightningCLI(save_config_kwargs={"overwrite": True, "config_filename": "config_override.yaml"})


if __name__ == "__main__":
    main()
