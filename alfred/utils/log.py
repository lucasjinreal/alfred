# -----------------------
#
# Copyright Jin Fagang @2018
# 
# 1/31/19
# log
# -----------------------
from loguru import logger
import sys


def init_logger():
    logger.remove()  # Remove the pre-configured handler
    logger.start(sys.stderr, format="<lvl>{level}</lvl> {time:MM-DD HH:mm:ss} {file}:{line} - {message}")

logger.remove()  # Remove the pre-configured handler
logger.start(sys.stderr, format="<green>{level}</green> <green>{time:MM.DD HH:mm:ss}</green> <blue>{file}:{line}:</blue> <green>{message}</green>")

