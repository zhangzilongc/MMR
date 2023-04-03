"""Argument parser functions."""

import argparse
import sys
import os

from config import get_cfg


def parse_args():
    """
    Parse the following arguments for a default parser.
    Args:
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide low level anomaly detection training and testing pipeline."
    )
    parser.add_argument(
        "--device",
        dest="device",
        help="the device to train model",
        default="0",
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_files",
        help="Path to the config files",
        default=["configs/R50.yaml"],
        nargs="+",
    )
    # add from command line
    parser.add_argument(
        "--opts",
        help="See config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args, path_to_config=None):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if path_to_config is not None:
        cfg.merge_from_file(path_to_config)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Create the checkpoint dir. ./checkpoints
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + "_" + str(cfg.RNG_SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return cfg
