
<div align="center">
<h1>StreamPETR with 3dppe Extension</h1>
</div>

## Introduction

This repository is an implementation of StreamPETR with 3dppe.

---
## Getting Started
1. Prepare [nuScenes dataset](https://www.nuscenes.org/download) and generate 2D annotations and temporal information for training & evaluation. (see [streamPETR](https://github.com/exiawsh/StreamPETR/blob/main/docs/data_preparation.md))
   
2. Conda env
```shell
conda create -n xxx python=3.8 -y
conda activate xxx
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install flash-attn==0.2.2  # (Tesla v100 is not compatible)

pip install mmcv-full==1.6.0
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc6 
pip install -v -e .
```

**Note** : make sure that 
`
numba 0.53.0
numpy 1.23.5
`   
(if not, reinstall numba==0.53.0).

**Catalogue**: 
`tree -d  -L 1  `
```
.
├── ckpts
├── data
├── mmdetection3d
├── projects
└── tools
```

3. Train & Infer
```bash
tools/dist_train.sh [-config] [-num_gpus]
```
```bash
tools/dist_test.sh [-config] [-model] [-num_gpus] --eval bbox
```

---
## Results on NuScenes Val Set

| Model | Setting |Pretrain| Lr Schd | Training Time | NDS| mAP | Config | Download|
| :---: | :---: | :---: | :---: | :---:|:---:| :---: | :---: |:---: |
|StreamPETR| V2-99 - 900q | [FCOS3D](https://github.com/exiawsh/storage/releases/download/v1.0/fcos3d_vovnet_imgbackbone-remapped.pth) | 24ep | 13h | 57.1 | 48.3 |[config](projects/configs/StreamPETR/stream_petr_vov_flash_800_bs2_seq_24e.py) |[model](https://github.com/exiawsh/storage/releases/download/v1.0/stream_petr_vov_flash_800_bs2_seq_24e.pth)/[log](https://github.com/exiawsh/storage/releases/download/v1.0/stream_petr_vov_flash_800_bs2_seq_24e.log) |
| **Stream3dppe** | V2-99 - 900q | [FCOS3D](https://github.com/exiawsh/storage/releases/download/v1.0/fcos3d_vovnet_imgbackbone-remapped.pth) | 24ep | 16h | **58.4** | **49.9** | [config](projects/configs/StreamPETR_3dppe/streampetr_3dppe_vov_flash_800_bs2_seq_24e_4x4_no_context_womv.py)| [model](https://drive.google.com/file/d/1Emyk0h2DK1mHuU8XGv5sVuCr1I4i7bAr/view?usp=sharing)/[log](https://drive.google.com/file/d/1hvo6y6uUVR8ixcZA9ktcCsYxEEi3KXDR/view?usp=drive_link) |
| **Stream3dppe_gt_detph** | V2-99 - 900q | [FCOS3D](https://github.com/exiawsh/storage/releases/download/v1.0/fcos3d_vovnet_imgbackbone-remapped.pth) | 24ep | 22h | 61.7 | 55.3 | [config](projects/configs/StreamPETR_3dppe/streampetr_3dppe_vov_flash_800_bs2_seq_24e_4x2_gtdepth.py)| [model](https://drive.google.com/file/d/1nNkIwY6rNFUlnXCkVF91l9sPyLQ77Chh/view?usp=drive_link)/[log](https://drive.google.com/file/d/1bzDzv4ErIbkvMcYxetDqnh70k2_yv_um/view?usp=drive_link) |

**Note :** `Stream3dppe` is trained on 4 x RTX 3090 with bs4 ,while `Stream3dppe_gt_detph` is trained on 4 x RTX 2080Ti with bs2 .

More result please refer to https://github.com/drilistbox/3DPPE.

---
## Acknowledgement
Many thanks to the authors of [PETR](https://github.com/megvii-research/PETR) and [StreamPETR](https://github.com/exiawsh/StreamPETR).

---
## Citation
If you find this project useful for your research, please consider citing: 
```bibtex   
@article{shu20233DPPE,
  title={3DPPE: 3D Point Positional Encoding for Multi-Camera 3D Object Detection Transformers},
  author={Shu, Changyong and Deng, Jiajun and Yu, Fisher and Liu, Yifan},
  journal={arXiv preprint arXiv:2211.14710},
  year={2023}
}
```