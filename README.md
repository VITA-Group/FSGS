# FSGS: Real-Time Few-Shot View Synthesis using Gaussian Splatting



[//]: # (###  [Project]&#40;https://zehaozhu.github.io/FSGS/&#41; | [Arxiv]&#40;https://arxiv.org/abs/2312.00451&#41;)

[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2312.00451)
[![Project Page](https://img.shields.io/badge/FSGS-Website-blue?logo=googlechrome&logoColor=blue)](https://zehaozhu.github.io/FSGS/)
[![Video](https://img.shields.io/badge/YouTube-Video-c4302b?logo=youtube&logoColor=red)](https://youtu.be/CarJgsx3DQY)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVITA-Group%2FFSGS&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)



---------------------------------------------------
<p align="center" >
  <a href="">
    <img src="https://github.com/zhiwenfan/zhiwenfan.github.io/blob/master/Homepage_files/videos/FSGS_gif.gif?raw=true" alt="demo" width="85%">
  </a>
</p>


## Environmental Setups
We provide install method based on Conda package and environment management:
```bash
conda env create --file environment.yml
conda activate FSGS
```
**CUDA 11.7** is strongly recommended.

## Data Preparation
In data preparation step, we reconstruct the sparse view inputs using SfM using the camera poses provided by datasets. Next, we continue the dense stereo matching under COLMAP with the function `patch_match_stereo` and obtain the fused stereo point cloud from `stereo_fusion`. 

``` 
cd FSGS
mkdir dataset 
cd dataset

# download LLFF dataset
gdown 16VnMcF1KJYxN9QId6TClMsZRahHNMW5g

# run colmap to obtain initial point clouds with limited viewpoints
python tools/colmap_llff.py

# download MipNeRF-360 dataset
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip -d mipnerf360 360_v2.zip

# run colmap on MipNeRF-360 dataset
python tools/colmap_360.py
``` 


We use the latest version of colmap to preprocess the datasets. If you meet issues on installing colmap, we provide a docker option. 
``` 
# if you can not install colmap, follow this to build a docker environment
docker run --gpus all -it --name fsgs_colmap --shm-size=32g  -v /home:/home colmap/colmap:latest /bin/bash
apt-get install pip
pip install numpy
python3 tools/colmap_llff.py
``` 


We provide both the sparse and dense point cloud after we proprecess them. You may download them [through this link](https://drive.google.com/drive/folders/1lYqZLuowc84Dg1cyb8ey3_Kb-wvPjDHA?usp=sharing). We use dense point cloud during training but you can still try sparse point cloud on your own.

## Training
Train FSGS on LLFF dataset with 3 views
``` 
python train.py  --source_path dataset/nerf_llff_data/horns --model_path output/horns --eval  --n_views 3 --sample_pseudo_interval 1
``` 


Train FSGS on MipNeRF-360 dataset with 24 views
``` 
python train.py  --source_path dataset/mipnerf360/garden --model_path output/garden  --eval  --n_views 24 --depth_pseudo_weight 0.03  
``` 


## Rendering
Run the following script to render the images.  

```
python render.py --source_path dataset/nerf_llff_data/horns/  --model_path  output/horns --iteration 10000
```

You can customize the rendering path as same as NeRF by adding `video` argument

```
python render.py --source_path dataset/nerf_llff_data/horns/  --model_path  output/horns --iteration 10000  --video  --fps 30
```

## Evaluation
You can just run the following script to evaluate the model.  

```
python metrics.py --source_path dataset/nerf_llff_data/horns/  --model_path  output/horns --iteration 10000
```

## Acknowledgement

Special thanks to the following awesome projects!

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [DreamGaussian](https://github.com/ashawkey/diff-gaussian-rasterization)
- [SparseNeRF](https://github.com/Wanggcong/SparseNeRF)
- [MipNeRF-360](https://github.com/google-research/multinerf)

## Citation
If you find our work useful for your project, please consider citing the following paper.


```
@misc{zhu2023FSGS, 
title={FSGS: Real-Time Few-Shot View Synthesis using Gaussian Splatting}, 
author={Zehao Zhu and Zhiwen Fan and Yifan Jiang and Zhangyang Wang}, 
year={2023},
eprint={2312.00451},
archivePrefix={arXiv},
primaryClass={cs.CV} 
}
```
