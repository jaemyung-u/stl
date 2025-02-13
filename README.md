<h1 align="center"> Self-supervised Transformation Learning for Equivariant Representations
</h1>

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2501.08712-b31b1b.svg)](https://arxiv.org/abs/2501.08712)&nbsp;

<div align="center">
  <a href="https://sites.google.com/view/jaemyungyu" target="_blank">Jaemyung&nbsp;Yu</a> &ensp; <b>&middot;</b> &ensp;
  <a href="https://sites.google.com/view/jaehyun-choi" target="_blank">Jaehyun&nbsp;Choi</a> &ensp; <b>&middot;</b> &ensp;
  Dong-Jae&nbsp;Lee &ensp; <b>&middot;</b> &ensp;
  HyeongGwon&nbsp;Hong &ensp; <b>&middot;</b> &ensp;
  <a href="http://siit.kaist.ac.kr/" target="_blank">Junmo&nbsp;Kim</a>
  <br>
  Korea Advanced Institute of Science and Technology (KAIST) &emsp; <br>
</div>
<h3 align="center">[<a href="https://jaemyung-u.github.io/stl/">project page</a>]&emsp;[<a href="https://arxiv.org/abs/2501.08712">arXiv</a>]</h3>
<br>

<b>Summary</b>: We propose Self-supervised Transformation Learning (STL), a method that learns transformation representations without explicit labels, enabling robust equivariant representation learning. By replacing transformation labels with self-supervised representations, STL captures interdependencies between transformations and handles complex augmentations like AugMix. STL improves generalization, achieving state-of-the-art classification accuracy on 7 out of 11 benchmarks and outperforming existing methods in object detection.

### 1. Setup

```bash
conda create -n stl python=3.10 -y
conda activate stl
pip install -r requirements.txt
```

### 2. Training

```bash
python main.py \
  --lr 0.03 \
  --weight-decay 5e-4 \
  --momentum 0.9 \
  --batch-size 256 \
  --projector 512-128 \
  --trans-backbone 128-128 \
  --trans-projector 128-128 \
  --inv 1.0 \
  --equi 1.0 \
  --trans 0.1 \
  --temperature 0.2 \
  --data-dir [DATA_PATH] \
  --save-dir [SAVE_PATH]
```

You can adjust the following options:

- `--inv`: Any positive values for invariance loss weight
- `--equi`: Any positive values for equivariance loss weight
- `--trans`: Any positive values for transformation loss weight

## Acknowledgement

This code is mainly built upon [AugSelf](https://github.com/hankook/AugSelf) and [SIE](https://github.com/facebookresearch/SIE) repositories.

## BibTeX

```bibtex
@inproceedings{yu2024stl,
  title={Self-supervised Transformation Learning for Equivariant Representations},
  author={Jaemyung Yu and Jaehyun Choi and Dong-Jae Lee and HyeongGwon Hong and Junmo Kim},
  year={2024},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
}
```