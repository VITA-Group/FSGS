import os
import numpy as np
import sys
import sqlite3

# os.system('python train.py  --source_path dataset/nerf_llff_data/fern --model_path output/fern --eval  --n_views 3 --sample_pseudo_interval 1')
# os.system('python train.py  --source_path dataset/nerf_llff_data/flower --model_path output/flower --eval  --n_views 3 --sample_pseudo_interval 1')
# os.system('python train.py  --source_path dataset/nerf_llff_data/horns --model_path output/horns --eval  --n_views 3 --sample_pseudo_interval 1')
# os.system('python train.py  --source_path dataset/nerf_llff_data/fortress --model_path output/fortress --eval  --n_views 3 --sample_pseudo_interval 1')
# os.system('python train.py  --source_path dataset/nerf_llff_data/leaves --model_path output/leaves --eval  --n_views 3 --sample_pseudo_interval 1')
# os.system('python train.py  --source_path dataset/nerf_llff_data/orchids --model_path output/orchids --eval  --n_views 3 --sample_pseudo_interval 1')
# os.system('python train.py  --source_path dataset/nerf_llff_data/room --model_path output/room --eval  --n_views 3 --sample_pseudo_interval 1')
# os.system('python train.py  --source_path dataset/nerf_llff_data/trex --model_path output/trex --eval  --n_views 3 --sample_pseudo_interval 1')


# os.system('python render.py --source_path dataset/nerf_llff_data/fern/  --model_path  output/fern --iteration 10000')
# os.system('python render.py --source_path dataset/nerf_llff_data/flower/  --model_path  output/flower --iteration 10000')
# os.system('python render.py --source_path dataset/nerf_llff_data/horns/  --model_path  output/horns --iteration 10000')
# os.system('python render.py --source_path dataset/nerf_llff_data/fortress/  --model_path  output/fortress --iteration 10000')
# os.system('python render.py --source_path dataset/nerf_llff_data/leaves/  --model_path  output/leaves --iteration 10000')
# os.system('python render.py --source_path dataset/nerf_llff_data/orchids/  --model_path  output/orchids --iteration 10000')
# os.system('python render.py --source_path dataset/nerf_llff_data/room/  --model_path  output/room --iteration 10000')
# os.system('python render.py --source_path dataset/nerf_llff_data/trex/  --model_path  output/trex --iteration 10000')


# os.system('python train.py --source_path dataset/nerf_synthetic/hotdog --model_path output/hotdog --eval --rand_pcd --n_views 8 --sample_pseudo_interval 1')
# for i in [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 10_00, 20_00, 30_00, 40_00, 50_00, 10_000]:
#     os.system('python render.py --source_path dataset/nerf_synthetic/hotdog/  --model_path  output/hotdog --iteration {}'.format(i))
os.system('python metrics_all.py --model_path  output/hotdog')