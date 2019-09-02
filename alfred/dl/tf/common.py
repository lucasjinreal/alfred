import os


def mute_tf():
    """
    this function will mute tensorflow
    disable tensorflow logging information
    call this before you import tensorflow
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'