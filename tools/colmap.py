import os
import numpy as np

def round_python3(number):
    rounded = round(number)
    if abs(number - rounded) == 0.5:
        return 2.0 * round(number / 2.0)
    return rounded

def pipeline(scene, base_path, n_views=3):
    view_path = str(n_views) + '_views'
    os.chdir(base_path + scene)
    os.system('rm -r ' + view_path)
    os.mkdir(view_path)
    os.chdir(view_path)
    os.mkdir('created')
    os.mkdir('triangulated')
    os.mkdir('images')
    os.system('colmap model_converter  --input_path ../sparse/0/ --output_path ../sparse/0/  --output_type TXT')

    images = {}
    with open('../sparse/0/images.txt', "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                fid.readline().split()
                images[image_name] = elems[1:]

    llffhold = 8
    img_list = sorted(images.keys(), key=lambda x: x)
    train_img_list = [c for idx, c in enumerate(img_list) if idx % llffhold != 0]
    if n_views > 0:
        idx_sub = [round_python3(i) for i in np.linspace(0, len(train_img_list)-1, n_views)]
        train_img_list = [c for idx, c in enumerate(train_img_list) if idx in idx_sub]


    for img_name in train_img_list:
        os.system('cp ../images/' + img_name + '  images/' + img_name)

    os.system('cp ../sparse/0/cameras.txt created/.')
    with open('created/points3D.txt', "w") as fid:
        pass

    res = os.popen('colmap feature_extractor --database_path database.db --image_path images  --SiftExtraction.max_image_size 4032 --SiftExtraction.max_num_features 32768 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1').read()
    os.system('colmap exhaustive_matcher --database_path database.db --SiftMatching.guided_matching 1 --SiftMatching.max_num_matches 32768')
    img_rank = [(name, res.find(name)) for name in train_img_list]
    img_rank = sorted(img_rank, key=lambda x: x[1])
    print(img_rank, res)
    with open('created/images.txt', "w") as fid:
        for idx, img_name in enumerate(img_rank):
            data = [str(1 + idx)] + [' ' + item for item in images[os.path.basename(img_name[0])]] + ['\n\n']
            fid.writelines(data)

    os.system('colmap point_triangulator --database_path database.db --image_path images --input_path created  --output_path triangulated  --Mapper.ba_local_max_num_iterations 40 --Mapper.ba_local_max_refinements 3 --Mapper.ba_global_max_num_iterations 100')
    os.system('colmap model_converter  --input_path triangulated --output_path triangulated  --output_type TXT')
    os.system('colmap image_undistorter --image_path images --input_path triangulated --output_path dense')
    os.system('colmap patch_match_stereo --workspace_path dense')
    os.system('colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply')



for scene in  ['fern', 'flower', 'fortress',  'horns',  'leaves',  'orchids',  'room',  'trex']:
    pipeline(scene, base_path = 'dataset/nerf_llff_data/', n_views = 3)
