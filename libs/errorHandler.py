import logging


def set_log(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
