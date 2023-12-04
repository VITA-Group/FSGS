# FSGS: Real-Time Few-Shot View Synthesis using Gaussian Splatting



###  [Project](https://zehaozhu.github.io/FSGS/) | [Arxiv](https://zehaozhu.github.io/FSGS/)



---------------------------------------------------


## Environmental Setups
We provide install method based on Conda package and environment management:
```bash
conda env create --file environment.yml
conda activate FSGS
```

## Data Preparation
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
In data preparation step, we reconstruct the sparse view inputs using SfM using the camera poses provided by datasets. Next, we continue the dense stereo matching under COLMAP with the function `patch_match_stereo` and obtain the fused stereo point cloud from `stereo_fusion`. 

## Training
To train FSGS on LLFF dataset with 3 views, please use 
``` 
python train.py  --source_path dataset/nerf_llff_data/horns --model_path output/horns --eval  --use_color --n_views 3 
``` 


To train FSGS on MipNeRF-360 dataset with 24 views, please use 
``` 
python train.py  --source_path dataset/mipnerf360/garden --model_path output/garden --eval  --use_color --n_views 24 
``` 


## Rendering
Run the following script to render the images.  

```
python render.py --source_path dataset/nerf_llff_data/horns/  --model_path  output/horns_full4 --iteration 10000  --video
```

You can customize the rendering path as same as NeRF by adding `video` argument

```
python render.py --source_path dataset/nerf_llff_data/horns/  --model_path  output/horns_full4 --iteration 10000  --video  --fps 30
```

## Evaluation
You can just run the following script to evaluate the model.  

```
python metrics.py --model_path "output/horns" 
```

## Acknowledgement

Special thanks to the [3D Gaussian Splatting](https://github.com/graphdeco-inria/diff-gaussian-rasterization) and [DreamGaussian](https://github.com/ashawkey/diff-gaussian-rasterization). 


## Citation
```

```