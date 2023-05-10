#!/usr/bin/env python3

"""Configs."""
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

_C.NUM_GPUS = 1
_C.RNG_SEED = 54

_C.OUTPUT_DIR = './log_low_level_AD'

_C.DATASET = CfgNode()
_C.DATASET.name = 'mvtec'
_C.DATASET.subdatasets = ["bottle",
                          "cable",
                          "capsule",
                          "carpet",
                          "grid",
                          "hazelnut",
                          "leather",
                          "metal_nut",
                          "pill",
                          "screw",
                          "tile",
                          "toothbrush",
                          "transistor",
                          "wood",
                          "zipper",
                          ]
_C.DATASET.resize = 256
# final image shape
_C.DATASET.imagesize = 224
_C.DATASET.domain_shift_category = "same"

_C.TRAIN = CfgNode()
_C.TRAIN.enable = True
_C.TRAIN.save_model = False
_C.TRAIN.method = 'PatchCore'
_C.TRAIN.backbone = 'resnet50'
_C.TRAIN.dataset_path = '/usr/sdd/zzl_data/MV_Tec'

# for MMR
_C.TRAIN.MMR = CfgNode()
_C.TRAIN.MMR.DA_low_limit = 0.2
_C.TRAIN.MMR.DA_up_limit = 1.
_C.TRAIN.MMR.layers_to_extract_from = ["layer1", "layer2", "layer3"]
# _C.TRAIN.MMR.freeze_encoder = False
_C.TRAIN.MMR.feature_compression = False
_C.TRAIN.MMR.scale_factors = (4.0, 2.0, 1.0)
_C.TRAIN.MMR.FPN_output_dim = (256, 512, 1024)
_C.TRAIN.MMR.load_pretrain_model = True
_C.TRAIN.MMR.model_chkpt = "./mae_visualize_vit_base.pth"
_C.TRAIN.MMR.finetune_mask_ratio = 0.75
_C.TRAIN.MMR.test_mask_ratio = 0.75

_C.TRAIN_SETUPS = CfgNode()
_C.TRAIN_SETUPS.batch_size = 64
_C.TRAIN_SETUPS.num_workers = 8
_C.TRAIN_SETUPS.learning_rate = 0.005
_C.TRAIN_SETUPS.epochs = 200
_C.TRAIN_SETUPS.weight_decay = 0.05
_C.TRAIN_SETUPS.warmup_epochs = 40

_C.TEST = CfgNode()
_C.TEST.enable = False
_C.TEST.method = 'PatchCore'
_C.TEST.save_segmentation_images = False
_C.TEST.save_video_segmentation_images = False
_C.TEST.dataset_path = '/usr/sdd/zzl_data/MV_Tec'

_C.TEST.VISUALIZE = CfgNode()
_C.TEST.VISUALIZE.Random_sample = True
_C.TEST.VISUALIZE.Sample_num = 40

# pixel auroc, aupro
_C.TEST.pixel_mode_verify = True

_C.TEST_SETUPS = CfgNode()
_C.TEST_SETUPS.batch_size = 64


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
