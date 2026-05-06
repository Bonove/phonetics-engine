import logging
import sys


def configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
