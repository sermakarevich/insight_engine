import argparse
import importlib
import logging
import sys

from app.logger import initialize_logging

logger = logging.getLogger(__name__)

PLUGINS = "crawlers"

def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Core parser")
    parser.add_argument("--plugin", type=str, help="Plugin to execute")
    return parser


def main() -> None:
    parser = get_argparser()
    args, _ = parser.parse_known_args()

    plugin_parser = getattr(importlib.import_module(f"{PLUGINS}.{args.plugin}.cli"), "get_argparser")
    plugin_kwargs = vars(plugin_parser().parse_known_args()[0])
    plugin_runner = getattr(importlib.import_module(f"{PLUGINS}.{args.plugin}.main"), "run")

    plugin_runner(**plugin_kwargs)


if __name__ == "__main__":
    initialize_logging()
    logger.info(f"Command-Line arguments: {' '.join(['python'] + sys.argv)}")
    main()
