from .utils import ForwardHook, cal_anomaly_map, each_patch_loss_function, mmr_adjust_learning_rate
from utils import compute_pixelwise_retrieval_metrics, compute_pro, save_image, save_video_segmentations

from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

import torch
import logging
import numpy as np

LOGGER = logging.getLogger(__name__)


class MMR_pipeline_:

    def __init__(self,
                 cur_model,
                 mmr_model,
                 optimizer,
                 device,
                 cfg):
        # register forward hook
        self.teacher_outputs_dict = {}
        for extract_layer in cfg.TRAIN.MMR.layers_to_extract_from:
            forward_hook = ForwardHook(self.teacher_outputs_dict, extract_layer)
            network_layer = cur_model.__dict__["_modules"][extract_layer]

            network_layer[-1].register_forward_hook(forward_hook)

        self.cur_model = cur_model.to(device)
        self.mmr_model = mmr_model.to(device)

        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg

    def fit(self, individual_dataloader):
        temporal_lr = self.cfg.TRAIN_SETUPS.learning_rate

        for epoch in range(self.cfg.TRAIN_SETUPS.epochs):
            self.cur_model.eval()
            self.mmr_model.train()
            current_lr = mmr_adjust_learning_rate(self.optimizer, epoch, self.cfg)
            if (epoch + 1) % 50 == 0:
                LOGGER.info("current lr is %.5f" % current_lr)

            loss_list = []

            for image in individual_dataloader:
                if isinstance(image, dict):
                    image = image["image"].to(self.device)
                else:
                    image = image.to(self.device)

                self.teacher_outputs_dict.clear()
                with torch.no_grad():
                    _ = self.cur_model(image)
                multi_scale_features = [self.teacher_outputs_dict[key]
                                        for key in self.cfg.TRAIN.MMR.layers_to_extract_from]
                reverse_features = self.mmr_model(image,
                                                  mask_ratio=self.cfg.TRAIN.MMR.finetune_mask_ratio)  # bn(inputs))
                multi_scale_reverse_features = [reverse_features[key]
                                                for key in self.cfg.TRAIN.MMR.layers_to_extract_from]

                loss = each_patch_loss_function(multi_scale_features, multi_scale_reverse_features)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())

            LOGGER.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1,
                                                            self.cfg.TRAIN_SETUPS.epochs,
                                                            np.mean(loss_list)))

        # reset learning rate
        self.cfg.TRAIN_SETUPS.learning_rate = temporal_lr

    def evaluation(self, test_dataloader):
        self.cur_model.eval()
        self.mmr_model.eval()

        labels_gt = []
        labels_prediction = []

        masks_gt = []
        masks_prediction = []
        aupro_list = []

        ima_path = []
        ima_name_list = []

        with torch.no_grad():
            for image in test_dataloader:
                if isinstance(image, dict):
                    label_current = image["is_anomaly"].numpy()
                    mask_current = image["mask"].squeeze(1).numpy()
                    labels_gt.extend(label_current.tolist())
                    masks_gt.extend(mask_current.tolist())

                    ima_path.extend(image["image_path"])
                    ima_name_list.extend(image["image_name"])

                    image = image["image"].to(self.device)
                else:
                    raise Exception("the format of DATA error!")

                self.teacher_outputs_dict.clear()
                with torch.no_grad():
                    _ = self.cur_model(image)
                multi_scale_features = [self.teacher_outputs_dict[key]
                                        for key in self.cfg.TRAIN.MMR.layers_to_extract_from]

                """
                try masking in test. Although it will produce higher abnormal scores, 
                but it simultaneously produce larger error for complex normal part or high variance area
                """
                reverse_features = self.mmr_model(image,
                                                  mask_ratio=self.cfg.TRAIN.MMR.test_mask_ratio)
                multi_scale_reverse_features = [reverse_features[key]
                                                for key in self.cfg.TRAIN.MMR.layers_to_extract_from]

                # return anomaly_map np.array (batch_size, imagesize, imagesize)
                anomaly_map, _ = cal_anomaly_map(multi_scale_features, multi_scale_reverse_features, image.shape[-1],
                                                 amap_mode='a')
                for item in range(len(anomaly_map)):
                    anomaly_map[item] = gaussian_filter(anomaly_map[item], sigma=4)

                labels_prediction.extend(np.max(anomaly_map.reshape(anomaly_map.shape[0], -1), axis=1))
                masks_prediction.extend(anomaly_map.tolist())

                # compute pro
                if self.cfg.TEST.pixel_mode_verify:
                    if set(mask_current.astype(int).flatten()) == {0, 1}:
                        aupro_list.extend(compute_pro(anomaly_map, mask_current.astype(int), label_current))

            auroc_samples = round(roc_auc_score(labels_gt, labels_prediction), 3)
            if self.cfg.TEST.pixel_mode_verify:
                pixel_scores = compute_pixelwise_retrieval_metrics(
                    masks_prediction, masks_gt
                )
                auroc_pixel = pixel_scores["auroc"]
            else:
                auroc_pixel = 0
                aupro_list = 0

            masks_prediction = np.stack(masks_prediction)

            """
            if normalizing the mask for each image, it will highlight the abnormal part, but it will
            hidden the effect in the normal image
            """
            # masks_prediction = min_max_norm(masks_prediction)

            if self.cfg.TEST.save_segmentation_images:
                save_image(cfg=self.cfg,
                           segmentations=masks_prediction,
                           masks_gt=masks_gt,
                           ima_path=ima_path,
                           ima_name_list=ima_name_list,
                           individual_dataloader=test_dataloader)

            if self.cfg.TEST.save_video_segmentation_images:
                save_video_segmentations(cfg=self.cfg,
                                         segmentations=masks_prediction,
                                         scores=np.array(labels_prediction),
                                         ima_path=ima_path,
                                         ima_name_list=ima_name_list,
                                         individual_dataloader=test_dataloader
                                         )

        return auroc_samples, auroc_pixel, round(np.mean(aupro_list), 3)

    def save_model(self):
        pass

    def load_model(self):
        pass
