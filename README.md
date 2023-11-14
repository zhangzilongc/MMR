## Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction

<p align="center">
  <img src=assets/image/mmr.png width="30%">
</p>

This is an official PyTorch implementation of the paper [Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction](https://arxiv.org/abs/2304.02216).
```
@article{zhang2023industrial,
  title={Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction},
  author={Zhang, Zilong and Zhao, Zhibin and Zhang, Xingwu and Sun, Chuang and Chen, Xuefeng},
  journal={arXiv preprint arXiv:2304.02216},
  year={2023}
}
```

### Datasets

We release a real-world Aero-engine Blade Anomaly Detection (AeBAD) dataset, consisting of two sub-datasets: the single-blade dataset (AeBAD-S) and the video anomaly detection dataset of blades (AeBAD-V). Compared to existing datasets, AeBAD has the following two characteristics: 1.) The target samples are not aligned and at different sacles. 2.) There is a domain shift between the distribution of normal samples in the test set and the training set, where the domain shifts are mainly caused by the changes in illumination and view.

**Download dataset at [here](https://drive.google.com/file/d/14wkZAFFeudlg0NMFLsiGwS0E593b-lNo/view?usp=share_link) (Google Drive) or [here](https://cloud.189.cn/web/share?code=nYraE3uMRJn2) (access code: g4pr) (Tian Yi Yun Pan).**

* AeBAD-S

<p align="center">
  <img src=assets/image/dataset_s.jpg width="80%">
</p>

* AeBAD-V

<p align="center">
  <img src=assets/image/dataset_v.jpg width="60%">
</p>

### Visualization for Videos

①: Original Video ②: PatchCore ③: ReverseDistillation ④: DRAEM ⑤: NSA ⑥: MMR

* video 1

<table rules="none" align="center">
	<tr>
        <td>
			<center>
				<img src=assets/video/video1/video_1_crop.gif width="100%" />
				<br/>
				<font color="AAAAAA">①</font>
			</center>
		</td>
		<td>
			<center>
				<img src=assets/video/video1/video_1_crop_PatchCore_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">②</font>
			</center>
		</td>
		<td>
			<center>
				<img src=assets/video/video1/video_1_crop_ReverseDistillation_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">③</font>
			</center>
		</td>
        <td>
			<center>
				<img src=assets/video/video1/video_1_crop_DRAEM_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">④</font>
			</center>
		</td>
        <td>
			<center>
				<img src=assets/video/video1/video_1_crop_NSA_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">⑤</font>
			</center>
		</td>
        <td>
			<center>
				<img src=assets/video/video1/video_1_crop_MMR_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">⑥</font>
			</center>
		</td>
	</tr>
</table>

* Video 2

<table rules="none" align="center">
	<tr>
        <td>
			<center>
				<img src=assets/video/video2/video_2_crop.gif width="100%" />
				<br/>
				<font color="AAAAAA">①</font>
			</center>
		</td>
		<td>
			<center>
				<img src=assets/video/video2/video_2_crop_PatchCore_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">②</font>
			</center>
		</td>
		<td>
			<center>
				<img src=assets/video/video2/video_2_crop_ReverseDistillation_process.gif width=100 />
				<br/>
				<font color="AAAAAA">③</font>
			</center>
		</td>
        <td>
			<center>
				<img src=assets/video/video2/video_2_crop_DRAEM_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">④</font>
			</center>
		</td>
        <td>
			<center>
				<img src=assets/video/video2/video_2_crop_NSA_process.gif width=100 />
				<br/>
				<font color="AAAAAA">⑤</font>
			</center>
		</td>
        <td>
			<center>
				<img src=assets/video/video2/video_2_crop_MMR_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">⑥</font>
			</center>
		</td>
	</tr>
</table>

* Video 3

<table rules="none" align="center">
	<tr>
        <td>
			<center>
				<img src=assets/video/video3/video_3_crop.gif width="100%" />
				<br/>
				<font color="AAAAAA">①</font>
			</center>
		</td>
		<td>
			<center>
				<img src=assets/video/video3/video_3_crop_PatchCore_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">②</font>
			</center>
		</td>
		<td>
			<center>
				<img src=assets/video/video3/video_3_crop_ReverseDistillation_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">③</font>
			</center>
		</td>
        <td>
			<center>
				<img src=assets/video/video3/video_3_crop_DRAEM_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">④</font>
			</center>
		</td>
        <td>
			<center>
				<img src=assets/video/video3/video_3_crop_NSA_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">⑤</font>
			</center>
		</td>
        <td>
			<center>
				<img src=assets/video/video3/video_3_crop_MMR_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">⑥</font>
			</center>
		</td>
	</tr>
</table>

### Get Started

#### Pre-trained models

Download the pre-trained model of MAE (ViT-base) at [here](https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth).

#### Dataset

**MVTec:**

**Create the MVTec dataset directory**. Download the MVTec-AD dataset from [here](https://www.mvtec.com/company/research/datasets/mvtec-ad). The MVTec dataset directory should be as follows. 

```
|-- data
    |-- MVTec-AD
        |-- mvtec_anomaly_detection
            |-- object (bottle, etc.)
                |-- train
                |-- test
                |-- ground_truth
```

**AeBAD:**

Download the AeBAD dataset from the above link. The AeBAD dataset directory should be as follows.

```
|-- AeBAD
    |-- AeBAD_S
        |-- train
            |-- good
                |-- background
        |-- test
                |-- ablation
                    |-- background
        |-- ground_truth
                |-- ablation
                    |-- view
    |-- AeBAD_V
        |-- test
            |-- video1
                |-- anomaly
        |-- train
            |-- good
                |-- video1_train
```

**Note that background, view and illumination in the train set is different from test. The background, view and illumination in test is unseen for the training set.**

#### Virtual Environment

Use the following commands:
```
pip install -r requirements.txt
```

#### Train and Test for MVTec, AeBAD

Train the model and evaluate it for each category or different domains. This will output the results (sample-level AUROC, pixel-level AUROC and PRO) for each category. It will generate the visualization in the directory.

run the following code:

```
sh mvtec_run.sh
```

```
sh AeBAD_S_run.sh
```

```
sh AeBAD_V_run.sh
```

TRAIN.MMR.model_chkpt in MMR.yaml is the path of above download model. TRAIN.dataset_path (TEST.dataset_path) is the path of data.
Set Test.save_segmentation_images as True or False to save processed image.

**Note that for AeBAD-V, we only evaluate the sample-level metric. The pixel-level metric is 0.**

## Acknowledgement
We acknowledge the excellent implementation from [MAE](https://github.com/facebookresearch/mae), [ViTDet](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet).

## License
The data is released under the CC BY 4.0 license.


