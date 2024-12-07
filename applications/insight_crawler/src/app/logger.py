import logging
import logging.config
import sys

STREAM_FORMAT = "%(levelname)-8s %(asctime)s %(name)s:%(lineno)s %(message)s"


def setup_stream_logger():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(STREAM_FORMAT))
    logging.getLogger().addHandler(handler)


def initialize_logging():
    setup_stream_logger()

    logging.getLogger().setLevel(logging.INFO)

    logging.getLogger("crawlers").setLevel(logging.INFO)
    logging.getLogger("__main__").setLevel(logging.INFO)
