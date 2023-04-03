import logging
import time
import sys
import os


def setup_logging(cfg):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    # Set up logging format.
    timeArray = time.localtime()
    otherStyleTime = time.strftime("%Y--%m--%d %H:%M", timeArray)
    otherStyleTime = otherStyleTime.replace(" ", "_").replace("--", "_").replace(":", "_")

    output_dir = cfg.OUTPUT_DIR
    OUTPUT_file_name = "_".join(["DATASET", cfg.DATASET.name,
                                 "METHOD", cfg.TRAIN.method,
                                 "RNG_SEED", str(cfg.RNG_SEED),
                                 "TIME", otherStyleTime])
    output_file_name = OUTPUT_file_name

    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(filename)s: %(lineno)4d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(plain_formatter)
    logger.addHandler(ch)

    filename = os.path.join(output_dir, output_file_name + '.log')

    file_info_handler = logging.FileHandler(filename)
    file_info_handler.setLevel(logging.INFO)
    file_info_handler.setFormatter(plain_formatter)

    logger.addHandler(file_info_handler)
