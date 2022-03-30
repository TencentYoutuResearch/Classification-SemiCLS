""" The Code is under Tencent Youtu Public Rule
"""
import logging
import os
import sys
import time
from datetime import datetime


def get_default_logger(
        args,
        logger_name,
        default_level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        rank=""):

    logger = logging.getLogger(logger_name)

    while not os.path.exists(args.out):
        time.sleep(0.1)

    logging.basicConfig(  # unlike the root logger, a custom logger can’t be configured using basicConfig()
        filename=os.path.join(args.out, f'{time_str()}_{rank}.log'),
        format=format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=default_level)
    logger.setLevel(default_level)

    # print
    # file_handler = logging.FileHandler()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(default_level)
    console_handler.setFormatter(logging.Formatter(format))
    logger.addHandler(console_handler)

    return logger


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'

    #     time.strftime(format[, t])
    return datetime.today().strftime(fmt)
