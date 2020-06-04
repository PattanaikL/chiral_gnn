import os
import logging


class Standardizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, rev=False):
        if rev:
            return (x * self.std) + self.mean
        return (x - self.mean) / self.std


def create_logger(name: str, log_dir: str = None) -> logging.Logger:
    """
    Creates a logger with a stream handler and file handler.

    :param name: The name of the logger.
    :param log_dir: The directory in which to save the logs.
    :return: The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Set logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fh = logging.FileHandler(os.path.join(log_dir, name + '.log'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    return logger