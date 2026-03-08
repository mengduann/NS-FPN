<p align="center"> <h1 align="center">Seeing Through the Noise: Improving Infrared Small Target Detection and Segmentation from Noise Suppression Perspective</h1>
  <p align="center">
    <b> CVPR, 2026 </b>
    <br />
    <a href="https://yuanmaoxun.github.io/"><strong>Maoxun Yuan </strong></a> 
    ·
    <a href=""><strong>Duanni Meng </strong></a>
    ·
    <a href=""><strong>Ziteng Xi </strong></a>
    ·
    <a href="https://github.com/Zhao-Tian-yi"><strong>Tianyi Zhao </strong></a>
    ·
    <a href="https://zhaoshiji123.github.io/"><strong>Shiji Zhao </strong></a>
      ·
    <a href="https://scholar.google.com.hk/citations?user=y5Ov6VAAAAAJ&hl=zh-CN&oi=ao"><strong>Yimian Dai </strong></a>
      ·
    <a href="https://sites.google.com/site/xingxingwei1988/"><strong>Xingxing Wei </strong></a>
  </p>


This repository is the official implementation of our paper [Seeing Through the Noise: Improving Infrared Small Target Detection and Segmentation from Noise Suppression Perspective](https://arxiv.org/html/2508.06878v2).

## Overview

<div align="center">
    <img src="./assets/NS-FPN.png" style="width: 80%; height: auto; max-height: 70vh;" alt="NS-FPN" />
</div>


## Introduction

Infrared small target detection and segmentation (IRSTDS) is a critical yet challenging task in defense and civilian applications, owing to the dim, shapeless appearance of targets and severe background clutter. Recent CNN-based methods have achieved promising target perception results, but they only focus on enhancing feature representation to offset the impact of noise, which results in the increased false alarm problem. In this paper, through analyzing the problem from the frequency domain, we pioneer in improving performance from noise suppression perspective and propose a novel noise-suppression feature pyramid network (NS-FPN), which integrates a low-frequency guided feature purification (LFP) module and a spiral-aware feature sampling (SFS) module into the original FPN structure. The LFP module suppresses the noise features by purifying high-frequency components to achieve feature enhancement devoid of noise interference, while the SFS module further adopts spiral sampling to fuse target-relevant features in feature fusion process. Our NS-FPN is designed to be lightweight yet effective and can be easily plugged into existing IRSTDS frameworks. Extensive experiments on the IRSTD-1k and NUAA-SIRST datasets demonstrate that our method significantly reduces false alarms and achieves superior performance on IRSTDS task.

## Quantitative Results

<div align="center">
    <img src="./assets/results.png" style="width: 80%; height: auto; max-height: 95vh;" alt="NS-FPN" />
</div>


## Visual Results

<div align="center">
    <img src="./assets/visualization.png" style="width: 90%; height: auto; max-height: 80vh;" alt="NS-FPN" />
</div>

## Usage
### Installation
1. Create and activate the conda environment:
```
conda env create -f environment.yml
```
2. Compile the SpiralFeatureSampling_MultiScaleDeformableAttention (A.K.A. SFS) module:
```
cd SFS_MSDeformAttn/ops/
sh make.sh
```
### Training
```python
python main.py --dataset-dir /path/to/Dataset --batch-size 16 --epochs 500 --lr 0.05 --mode train --warm-epoch 5
```
### Testing
```python
python main.py --dataset-dir /path/to/Dataset --batch-size 1 --mode test --weight-path /path/to/weight.pkl
```
### Best Weights
| Dataset         | IoU (x10(-2)) | Pd (x10(-2))|  Fa (x10(-6)) | Download |
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| IRSTD-1k | 69.34 | 95.58 | 8.35 | [IRSTD-1k_weights](https://drive.google.com/file/d/1agnCjpJJa3J3-Aw8XqDKtpcTA6xDuHO4/view?usp=sharing) |
| NUAA-SIRST | 78.74 | 100.0 | 1.24 | [NUAA-SIRST_weights](https://drive.google.com/file/d/17zgfkbkPdLGyOLDz_MFNbUQI9J2bmgiI/view?usp=sharing) |
* Our NS-FPN is developed based on [MSHNet](https://github.com/Lliu666/MSHNet). Thanks to Qiankun Liu.

## Citation

If you find this code useful for your research, please consider citing:

```bibtex
@article{yuan2025ns,
  title={NS-FPN: Improving infrared small target detection and segmentation from noise suppression perspective},
  author={Yuan, Maoxun and Meng, Duanni and Xi, Ziteng and Zhao, Tianyi and Zhao, Shiji and Dai, Yimian and Wei, Xingxing},
  journal={arXiv preprint arXiv:2508.06878},
  year={2025}
}
