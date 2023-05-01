"""coin-test backtesting library."""

import logging

from .orchestration import run

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.2.0"
__all__ = ["run"]
