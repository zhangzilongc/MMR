import random

import numpy as np
import torch
import logging

from utils import get_dataloaders, load_backbones
from utils.common import freeze_paras, scratch_MAE_decoder

from models.MMR import MMR_base, MMR_pipeline_

import timm.optim.optim_factory as optim_factory

LOGGER = logging.getLogger(__name__)


def train(cfg=None):
    """
    include data loader load, model load, optimizer, training and test.
    """
    # Set random seed from configs.
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)

    LOGGER.info("load dataset!")
    # get train dataloader (include each category)
    train_dataloaders = get_dataloaders(cfg=cfg, mode='train')
    # get test dataloader (include each category)
    test_dataloaders = get_dataloaders(cfg=cfg, mode='test')

    cur_device = torch.device("cuda:0")

    result_collect = {"AUROC": [],
                      "Pixel-AUROC": [],
                      "per-region-overlap (PRO)": []}

    # training process
    for idx, individual_dataloader in enumerate(train_dataloaders):
        LOGGER.info("current individual_dataloader is {}.".format(individual_dataloader.name))
        LOGGER.info("the data in current individual_dataloader {} are {}.".format(individual_dataloader.name,
                                                                                  len(individual_dataloader.dataset)))

        # load model
        if cfg.TRAIN.method in ['MMR']:
            # target model
            cur_model = load_backbones(cfg.TRAIN.backbone)
            freeze_paras(cur_model)

            # mask model prepare
            mmr_base = MMR_base(cfg=cfg,
                                scale_factors=cfg.TRAIN.MMR.scale_factors,
                                FPN_output_dim=cfg.TRAIN.MMR.FPN_output_dim)

            if cfg.TRAIN.MMR.load_pretrain_model:
                checkpoint = torch.load(cfg.TRAIN.MMR.model_chkpt)
                checkpoint = scratch_MAE_decoder(checkpoint)
                LOGGER.info("train the decoder FPN of MMR from scratch!")

                msg = mmr_base.load_state_dict(checkpoint['model'], strict=False)
                LOGGER.info("MAE load meg: {}".format(msg))
            else:
                LOGGER.info("MAE train from scratch!")
        else:
            raise NotImplementedError("train method {} does not include in target methods".format(cfg.TRAIN.method))

        # optimizer load
        optimizer = None
        if cfg.TRAIN.method in ['MMR']:
            # following timm: set wd as 0 for bias and norm layers (AdamW)
            param_groups = optim_factory.add_weight_decay(mmr_base, cfg.TRAIN_SETUPS.weight_decay)
            optimizer = torch.optim.AdamW(param_groups, lr=cfg.TRAIN_SETUPS.learning_rate, betas=(0.9, 0.95))
        else:
            raise NotImplementedError("train method {} does not include in target methods".format(cfg.TRAIN.method))

        # start training
        torch.cuda.empty_cache()
        if cfg.TRAIN.method == 'MMR':
            MMR_instance = MMR_pipeline_(cur_model=cur_model,
                                            mmr_model=mmr_base,
                                            optimizer=optimizer,
                                            device=cur_device,
                                            cfg=cfg)
            MMR_instance.fit(individual_dataloader)
        else:
            raise NotImplementedError("train method {} does not include in target methods".format(cfg.TRAIN.method))

        torch.cuda.empty_cache()
        LOGGER.info("current test individual_dataloader is {}.".format(test_dataloaders[idx].name))
        LOGGER.info("the test data in current individual_dataloader {} are {}.".format(test_dataloaders[idx].name,
                                                                                       len(test_dataloaders[
                                                                                               idx].dataset)))
        LOGGER.info("Computing evaluation metrics.")
        """
                            prediction
                        ______1________0____
                      1 |    TP   |   FN   |
        ground truth  0 |    FP   |   TN   |

        ACC = (TP + TN) / (TP + FP + FN + TN)

        precision = TP / (TP + FP)

        recall (TPR) = TP / (TP + FN)

        FPR（False Positive Rate）= FP / (FP + TN)
        """
        if cfg.TRAIN.method == 'MMR':
            auc_sample, auroc_pixel, pro_auc = MMR_instance.evaluation(
                test_dataloader=test_dataloaders[idx])
        else:
            raise NotImplementedError("train method {} does not include in target methods".format(cfg.TRAIN.method))

        result_collect["AUROC"].append(auc_sample)
        LOGGER.info("{}'s Image_Level AUROC is {:2f}.%".format(individual_dataloader.name, auc_sample * 100))

        result_collect["Pixel-AUROC"].append(auroc_pixel)
        LOGGER.info(
            "{}'s Full_Pixel_Level AUROC is {:2f}.%".format(individual_dataloader.name, auroc_pixel * 100))

        result_collect["per-region-overlap (PRO)"].append(pro_auc)
        LOGGER.info(
            "{}'s per-region-overlap (PRO) AUROC is {:2f}.%".format(individual_dataloader.name, pro_auc * 100))

    LOGGER.info("Method training phase complete!")

    for key, values in result_collect.items():
        LOGGER.info(
            "Mean {} is {:2f}.%".format(key, np.mean(np.array(values)) * 100))
