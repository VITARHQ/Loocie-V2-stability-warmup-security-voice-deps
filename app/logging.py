import logging
import sys

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
