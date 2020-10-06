# Scribble2Label: Scribble-Supervised Cell Segmentation via Self-Generating Pseudo-Labels with Consistency
This is official PyTorch implementation of Scribble2Label (MICCAI 2020). For technical details, please refer to:
___
**Scribble2Label: Scribble-Supervised Cell Segmentation via Self-Generating Pseudo-Labels with Consistency**

[Hyeonsoo Lee](https://scholar.google.com/citations?user=BV-AwjoAAAAJ&hl=ko&authuser=2), [Won-ki Jeong](https://scholar.google.co.kr/citations?user=bnyKqkwAAAAJ&hl=ko)

**[[Paper](https://arxiv.org/abs/2006.12890)]**

**MICCAI 2020**

![Overview](/figure/Overview.png)

- Segmentation is a fundamental process in microscopic cell image analysis.
With the advent of recent advances in deep learning, more accurate and high-throughput cell segmentation has become feasible.
However, most existing deep learning-based cell segmentation algorithms require fully annotated ground-truth cell labels, which are time-consuming and labor-intensive to generate.
In this paper, we introduce Scribble2Label, a novel weakly-supervised cell segmentation framework that exploits only a handful of scribble annotations without full segmentation labels.
The core idea is to combine pseudo-labeling and label filtering to generate reliable labels from weak supervision. For this, we leverage the consistency of predictions by iteratively averaging the predictions to improve pseudo labels.
We demonstrate the performance of Scribble2Label by comparing it to several state-of-the-art cell segmentation methods with various cell image modalities, including bright-field, fluorescence, and electron microscopy.
We also show that our method performs robustly across different levels of scribble details, which confirms that only a few scribble annotations are required in real-use cases.

___
## License

```
Copyright (c) 2020, Korea University. Hyeonsoo Lee, Won-ki Jeong.
All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Permission to use, copy, modify, and distribute this software and its documentation
for any non-commercial purpose is hereby granted without fee, provided that the above
copyright notice appear in all copies and that both that copyright notice and this
permission notice appear in supporting documentation, and that the name of the author
not be used in advertising or publicity pertaining to distribution of the software
without specific, written prior permission.

THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL
DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
```

## Environment Setup

Code has been run and tested on Ubuntu 18.04, Python 3.7, Pytorch 1.5, CUDA 10.0, TITAN X and GTX 1080Ti GPUs.
- **Install Python Packages**
```shell script
pip install -r requirements.txt
```
- **Download and Preprocess Dataset (BBBC038v1)**
    - Shell Script includes these procedures:
    1. Download BBBC038v1 Dataset from [the official website](https://bbbc.broadinstitute.org/BBBC038).
    2. Cluster Fluorescence, Histopathology, Bright Field images in the dataset.
    3. Generate Scribble labels automatically.
    4. Split Training & Test Data of each modalities.
```shell script
/bin/bash preprocess_dataset.sh
```
- **Training**

Train Scirbble2Label on BBBC038v1 dataset.
```shell script
python Train.py
```
Modalities/Scribble Details/Hyperparameters can be controlled by changing config class in *Train.py*.
Default setting is Fluorescence/30% Scribbles.

- **Inference**

Inference Test Data on BBBC038v1 dataset.
```shell script
python Inference.py
```
Modalities/Scribble Details/Hyperparameters can be controlled by changing config class in *Inference.py*.

- **Pretrained Weights**

|DSB-Fluo 10%|DSB-Fluo 30%|DSB-Fluo 50%|DSB-Fluo 100%|
|:---:|:---:|:---:|:---:|
|[link](https://drive.google.com/file/d/11vWtzi9ippVeGnerW2X1-6tTJt9pdY_u/view?usp=sharing)|[link](https://drive.google.com/file/d/1y8EtLGaEL-tTAjVfGJgy2sRlJkIxUog6/view?usp=sharing)|[link](https://drive.google.com/file/d/1BuyOSrWC7QdlsTXoH2KAIrXL0sxVlDGS/view?usp=sharing)|[link](https://drive.google.com/file/d/1UNrl1p4Z4t05lf7q_zo6XSN9S0-LLS3-/view?usp=sharing)|

## Citation
If you use this code, please cite:
```
@inproceedings{scribble2label,
title={Scribble2Label: Scribble-Supervised Cell Segmentation via Self-Generating Pseudo-Labels with Consistency},
author={Hyeonsoo Lee and Won-ki Jeong},
booktitle={MICCAI},
year={2020}
}
```
