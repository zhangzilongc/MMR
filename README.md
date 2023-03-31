## Masked Multi-scale Reconstruction for Real-world Industrial Anomaly Detection with Domain Shift

<p align="center">
  <img src=assets/image/mmr.png width="30%">
</p>

This is an official PyTorch implementation of the paper Masked Multi-scale Reconstruction for Real-world Industrial Anomaly Detection with Domain Shift.

### Datasets

We release a real Aero-engine Blade Anomaly Detection (AeBAD) dataset, consisting of two sub-datasets: the single-blade dataset (AeBAD-S) and the video anomaly detection dataset of blades (AeBAD-V). Compared to existing datasets, AeBAD has the following two characteristics: 1.) The target samples are not aligned and at different sacles. 2.) There is a domain shift between the distribution of normal samples in the test set and the training set, where the domain shifts are mainly caused by the changes in illumination and view.

**Dataset will be available soon.**

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

#### Environment

#### Train

#### Test





