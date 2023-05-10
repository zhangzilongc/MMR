import numpy as np
from numpy import ndarray
from sklearn import metrics
import cv2
import os
import pandas as pd
from skimage import measure
from statistics import mean
from sklearn.metrics import auc
import random
from torchvision.datasets.folder import default_loader
import logging
import math

from torchvision import transforms as T
from PIL import Image

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch

LOGGER = logging.getLogger(__name__)


def seed_everything(seed):
    torch.manual_seed(seed)  # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False  # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)


def freeze_paras(backbone):
    for para in backbone.parameters():
        para.requires_grad = False


def freeze_MAE_paras(MAE_model):
    for name, param in MAE_model.named_parameters():
        if "decoder" not in name and name != "mask_token":
            param.requires_grad = False


def scratch_MAE_decoder(checkpoint):
    for key_indv in list(checkpoint["model"].keys()):
        if "decoder" in key_indv or key_indv == "mask_token":
            checkpoint["model"].pop(key_indv)
    return checkpoint


def compute_imagewise_retrieval_metrics(
        anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    # flatten
    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    mean_AP = metrics.average_precision_score(flat_ground_truth_masks.astype(int),
                                              flat_anomaly_segmentations)

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
        "mean_AP": mean_AP
    }


def compute_pro(anomaly_map: ndarray, gt_mask: ndarray, label: ndarray, num_th: int = 200):
    assert isinstance(anomaly_map, ndarray), "type(amaps) must be ndarray"
    assert isinstance(gt_mask, ndarray), "type(masks) must be ndarray"
    assert anomaly_map.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert gt_mask.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert anomaly_map.shape == gt_mask.shape, "amaps.shape and masks.shape must be same"
    assert set(gt_mask.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    current_amap = anomaly_map[label != 0]
    current_mask = gt_mask[label != 0].astype(int)

    binary_amaps = np.zeros_like(current_amap[0], dtype=np.bool)
    pro_auc_list = []

    for anomaly_mask, mask in zip(current_amap, current_mask):
        df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
        min_th = anomaly_mask.min()
        max_th = anomaly_mask.max()
        delta = (max_th - min_th) / num_th

        for th in np.arange(min_th, max_th, delta):
            binary_amaps[anomaly_mask <= th] = 0
            binary_amaps[anomaly_mask > th] = 1

            pros = []
            # for connect region
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amaps[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

            inverse_masks = 1 - mask
            fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()

            fpr = fp_pixels / inverse_masks.sum()

            df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

        # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
        df = df[df["fpr"] < 0.3]
        df["fpr"] = df["fpr"] / df["fpr"].max()

        pro_auc = auc(df["fpr"], df["pro"])

        pro_auc_list.append(pro_auc)

    return pro_auc_list


def save_image(cfg, segmentations: ndarray, masks_gt, ima_path, ima_name_list, individual_dataloader):
    """
    segmentations: normalized segmentations.

    add mask_AD pred mask
    """
    save_fig_path = os.path.join(cfg.OUTPUT_DIR, "image_save")
    os.makedirs(save_fig_path, exist_ok=True)

    sample_num = len(segmentations)

    segmentations_max, segmentations_min = np.max(segmentations), np.min(segmentations)

    # visualize for random sample
    if cfg.TEST.VISUALIZE.Random_sample:
        sample_idx = random.sample(range(sample_num), cfg.TEST.VISUALIZE.Sample_num)
    else:
        sample_idx = [i for i in range(sample_num)]

    segmentations_random_sample = [segmentations[idx_random] for idx_random in sample_idx]
    mask_random_sample = [masks_gt[idx_random] for idx_random in sample_idx]
    ima_path_random_sample = [ima_path[idx_random] for idx_random in sample_idx]
    ima_name_random_sample = [ima_name_list[idx_random] for idx_random in sample_idx]

    temp_individual_name = os.path.join(save_fig_path, individual_dataloader.name)
    os.makedirs(temp_individual_name, exist_ok=True)

    for idx, (seg_each, mask_each, ori_path_each, name_each) in enumerate(zip(segmentations_random_sample,
                                                                              mask_random_sample,
                                                                              ima_path_random_sample,
                                                                              ima_name_random_sample)):
        anomaly_type = name_each.split("/")[2]
        temp_anomaly_name = os.path.join(temp_individual_name, anomaly_type)
        os.makedirs(temp_anomaly_name, exist_ok=True)
        file_name = name_each.replace("/", "_").split(".")[0]

        mask_numpy = np.squeeze((255 * np.stack(mask_each)).astype(np.uint8))

        original_ima = individual_dataloader.dataset.transform_mask(default_loader(ori_path_each))
        original_ima = (original_ima.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        original_ima = cv2.cvtColor(original_ima, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(11, 10))
        sns.heatmap(seg_each, vmin=segmentations_min, vmax=segmentations_max, xticklabels=False,
                    yticklabels=False, cmap="jet", cbar=True)
        plt.savefig(os.path.join(temp_anomaly_name, f'{file_name}_sns_heatmap.jpg'),
                    bbox_inches='tight', pad_inches=0.0)
        plt.close()

        # min-max normalize for all images
        seg_each = (seg_each - segmentations_min) / (segmentations_max - segmentations_min)

        # only for seg_each that range in (0, 1)
        seg_each = np.clip(seg_each * 255, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(seg_each, cv2.COLORMAP_JET)

        if heatmap.shape != original_ima.shape:
            raise Exception("ima shape is not consistent!")

        heatmap_on_image = np.float32(heatmap) / 255 + np.float32(original_ima) / 255
        heatmap_on_image = heatmap_on_image / np.max(heatmap_on_image)
        heatmap_on_image = np.uint8(255 * heatmap_on_image)

        cv2_ima_save(temp_anomaly_name,
                     file_name,
                     ori_ima=original_ima,
                     mask_ima=mask_numpy,
                     heat_ima=heatmap,
                     heat_on_ima=heatmap_on_image)
    LOGGER.info("image save complete!")


def save_video_segmentations(cfg, segmentations: ndarray, scores: ndarray, ima_path, ima_name_list,
                             individual_dataloader):
    save_fig_path = os.path.join(cfg.OUTPUT_DIR, "video_save")
    os.makedirs(save_fig_path, exist_ok=True)

    sample_num = len(segmentations)

    # obtain the max segmentations
    segmentations_max, segmentations_min = np.max(segmentations), np.min(segmentations)

    sample_idx = [i for i in range(sample_num)]

    segmentations_random_sample = [segmentations[idx_random] for idx_random in sample_idx]
    scores = scores.tolist()
    ima_path_random_sample = [ima_path[idx_random] for idx_random in sample_idx]
    ima_name_random_sample = [ima_name_list[idx_random] for idx_random in sample_idx]

    temp_individual_name = os.path.join(save_fig_path, individual_dataloader.name)
    os.makedirs(temp_individual_name, exist_ok=True)

    for seg_each, score_each, ori_path_each, name_each in zip(segmentations_random_sample,
                                                              scores,
                                                              ima_path_random_sample,
                                                              ima_name_random_sample):
        anomaly_type = name_each.split("/")[1]
        temp_anomaly_name = os.path.join(temp_individual_name, anomaly_type)
        os.makedirs(temp_anomaly_name, exist_ok=True)
        file_name = name_each.replace("/", "_").split(".")[0]

        original_ima = individual_dataloader.dataset.transform_mask(default_loader(ori_path_each))
        original_ima = (original_ima.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        original_ima = cv2.cvtColor(original_ima, cv2.COLOR_BGR2RGB)

        seg_each = (seg_each - segmentations_min) / (segmentations_max - segmentations_min)

        seg_each = np.clip(seg_each * 255, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(seg_each, cv2.COLORMAP_JET)

        if heatmap.shape != original_ima.shape:
            raise Exception("ima shape is not consistent!")

        heatmap_on_image = np.float32(heatmap) / 255 + np.float32(original_ima) / 255
        heatmap_on_image = heatmap_on_image / np.max(heatmap_on_image)
        heatmap_on_image = np.uint8(255 * heatmap_on_image)

        str_score_each = str(score_each).replace(".", "_")

        cv2.imwrite(os.path.join(temp_anomaly_name, f'{file_name}_heatmap_{str_score_each}.jpg'), heatmap_on_image)
    LOGGER.info("image save complete!")


def cv2_ima_save(dir_path, file_name, ori_ima, mask_ima, heat_ima, heat_on_ima):
    cv2.imwrite(os.path.join(dir_path, f'{file_name}_original.jpg'), ori_ima)
    cv2.imwrite(os.path.join(dir_path, f'{file_name}_mask.jpg'), mask_ima)
    cv2.imwrite(os.path.join(dir_path, f'{file_name}_heatmap.jpg'), heat_ima)
    cv2.imwrite(os.path.join(dir_path, f'{file_name}_hm_on_ima.jpg'), heat_on_ima)
