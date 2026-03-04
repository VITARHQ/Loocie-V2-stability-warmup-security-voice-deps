import logging
import json
import sys
from datetime import datetime, timezone


class HumanFormatter(logging.Formatter):
    GREY   = "\x1b[38;5;240m"
    BLUE   = "\x1b[36m"
    RESET  = "\x1b[0m"

    LEVEL_COLORS = {
        "DEBUG":    "\x1b[38;5;240m",
        "INFO":     "\x1b[36m",
        "WARNING":  "\x1b[33m",
        "ERROR":    "\x1b[31m",
        "CRITICAL": "\x1b[1m\x1b[31m",
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelname, self.RESET)
        time  = datetime.now(timezone.utc).strftime("%H:%M:%S")
        level = record.levelname.ljust(8)
        return f"{self.GREY}{time}{self.RESET}  {color}{level}{self.RESET}  {self.BLUE}{record.name}{self.RESET}  {record.getMessage()}"


def setup_logging(level="INFO"):
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(HumanFormatter())
    console.setLevel(numeric_level)
    root = logging.getLogger("loocie")
    root.setLevel(numeric_level)
    root.handlers.clear()
    root.addHandler(console)
    root.propagate = False
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)
    root.info("[LOOCIE] Logging initialized - level=%s", level.upper())


def get_logger(name):
    return logging.getLogger(f"loocie.{name}")
