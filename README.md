
<div align="center">
<h1>StreamPETR with 3dppe Extension</h1>
</div>

## Introduction

This repository is an implementation of StreamPETR with 3dppe.

---
## Getting Started
1. Prepare [nuScenes dataset](https://www.nuscenes.org/download) and g2D annotations and temporal information for training/evaluation. (see [streamPETR](https://github.com/exiawsh/StreamPETR/blob/main/docs/data_preparation.md))
   
2. Conda env :
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
numba                   0.53.0
numpy                   1.23.5
`
(if not, reinstall numba==0.53.0)

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
## Results on NuScenes Val Set.

| Model | Setting |Pretrain| Lr Schd | Training Time | NDS| mAP | Config | Download|
| :---: | :---: | :---: | :---: | :---:|:---:| :---: | :---: |:---: |
| stream3dppe| V2-99 - 900q | [FCOS3D](https://github.com/exiawsh/storage/releases/download/v1.0/fcos3d_vovnet_imgbackbone-remapped.pth) | 24ep | 16h | 58.19 | 49.75 | [config](projects/configs/StreamPETR_3dppe/streampetr_3dppe_vov_flash_800_bs2_seq_24e_4x4_no_context_womv.py)| [model]()/[log]() |
| stream3dppe_gt_detph| V2-99 - 900q | [FCOS3D](https://github.com/exiawsh/storage/releases/download/v1.0/fcos3d_vovnet_imgbackbone-remapped.pth) | 24ep | 22h | 61.78 | 55.30 | [config](projects/configs/StreamPETR_3dppe/streampetr_3dppe_vov_flash_800_bs2_seq_24e_4x2_gtdepth.py)| [model]()/[log](https://www.aliyundrive.com/s/DSY1hbfgB5P) |
