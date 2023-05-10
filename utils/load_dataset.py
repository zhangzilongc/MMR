import logging
import torch
from enum import Enum


LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["datasets.mvtec", "MVTecDataset"],
             "aebad_S": ["datasets.aebad_S", "AeBAD_SDataset"],
             "aebad_V": ["datasets.aebad_V", "AeBAD_VDataset"]
             }


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def get_dataloaders(cfg, mode='train'):
    dataset_info = _DATASETS[cfg.DATASET.name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    # dataloaders contain different objects (bottle, screw, etc.)
    dataloaders = []
    shuffle = False if cfg.TRAIN.method in ['PatchCore'] else True

    # cfg.DATASET.subdatasets is a list which includes diverse objects
    for subdataset in cfg.DATASET.subdatasets:
        dataset = dataset_library.__dict__[dataset_info[1]](
            source=cfg.TRAIN.dataset_path if mode == 'train' else cfg.TEST.dataset_path,
            classname=subdataset,
            resize=cfg.DATASET.resize,
            imagesize=cfg.DATASET.imagesize,
            split=DatasetSplit.TRAIN if mode == 'train' else DatasetSplit.TEST,
            cfg=cfg,
            seed=cfg.RNG_SEED
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.TRAIN_SETUPS.batch_size if mode == 'train' else cfg.TEST_SETUPS.batch_size,
            shuffle=shuffle if mode == 'train' else False,
            num_workers=cfg.TRAIN_SETUPS.num_workers,
            pin_memory=True,
        )

        dataloader.name = cfg.DATASET.name
        if subdataset is not None:
            dataloader.name += "_" + subdataset

        dataloaders.append(dataloader)

    return dataloaders
