#!/usr/bin/python3

import logging
import os
import pprint

from utils import setup_logging, load_config, parse_args

from tools import train

LOGGER = logging.getLogger(__name__)


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    for path_to_config in args.cfg_files:
        # merge config and args, mkdir image_save and checkpoints
        cfg = load_config(args, path_to_config=path_to_config)
        # Setup logging format.
        setup_logging(cfg)
        LOGGER.info(pprint.pformat(cfg))
        LOGGER.info("path_to_config is {}".format(path_to_config))

        # Perform training and test in each category.
        if cfg.TRAIN.enable:
            LOGGER.info("start training!")
            """
            include:
             1) train and test dataloader load
             2) training prepare phase: 1) base model load
                                        2) optimizer load
             3) start training: include various methods (class module)
             4) complete training: start test (one follow by one)
            """
            train(cfg=cfg)

        LOGGER.info("Main function complete!")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
