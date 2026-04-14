"""
Command-line interface for Aura.

Entry point for high-throughput Rietveld analysis workflows.

Example usage:
    $ aura --help
    $ aura --version
"""

import click

from aura import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """Aura - Prototype framework for large-volume Rietveld analysis."""


if __name__ == "__main__":
    main()
