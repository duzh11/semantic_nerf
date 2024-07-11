import os
import cv2

from glob import glob
from tqdm import tqdm
import numpy as np

scene_lis = ['scene0050_00', 'scene0616_00', 'scene0169_00', 'scene0426_00', \
             'scene0084_00', 'scene0025_00', ]
snerf_exp_root = '/home/du/Proj/3Dv_Semantics/semantic_nerf/exps'


for scene in tqdm(scene_lis, desc='viz depth'):
    snerf_scene_exp = os.path.join(snerf_exp_root, scene+'_NeuRIS')
    snerf_scene_depth_dir = os.path.join(snerf_scene_exp, 'train_render', 'step_200000')
    snerf_scene_depth_lis = sorted(glob(os.path.join(snerf_scene_depth_dir, 'depth_*.png')))

    for depth_file_idx in snerf_scene_depth_lis:
        depth_idx = cv2.imread(depth_file_idx, cv2.IMREAD_UNCHANGED)
        depth_idx = (depth_idx.astype(np.float32)/1000.0)*50

        depth_vis = cv2.convertScaleAbs(depth_idx)
        depth_vis_jet = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imwrite(f'{snerf_scene_depth_dir}/' + os.path.basename(depth_file_idx)[:-4]+'_jet.png', depth_vis_jet)