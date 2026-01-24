"""Logging infrastructure for FastHarness."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a FastHarness module.

    Args:
        name: Module name (e.g., "worker", "client")

    Returns:
        Logger with fastharness.{name} prefix
    """
    return logging.getLogger(f"fastharness.{name}")
