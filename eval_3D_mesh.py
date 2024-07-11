import os, shutil
import json
import cv2
import numpy as np  
from glob import glob
from tqdm import tqdm

import utils_evaulation as EvauationUtils

scene_lis = ['scene0378_00', 'scene0435_02', 'scene0050_00', 'scene0616_00', \
             'scene0084_00', 'scene0169_00', 'scene0025_00', 'scene0426_00']
level=0.6

neuris_data_root = '/home/du/Proj/3Dv_Reconstruction/NeuRIS/Data/dataset/indoor'
snerf_data_root = '/home/du/Proj/3Dv_Semantics/semantic_nerf/Data'
snerf_exp_root = '/home/du/Proj/3Dv_Semantics/semantic_nerf/exps'

metric_all = []
for scene in tqdm(scene_lis, desc='Evaluating 2D Semantic'):
    neuris_scene_root = os.path.join(neuris_data_root, scene)
    snerf_scene_exp = os.path.join(snerf_exp_root, scene+'_NeuRIS')
    
    dir_scan = neuris_scene_root
    target_img_size = (640, 480)
    # contrust TSDF of ground truth mesh   
    path_mesh_gt = f'{neuris_scene_root}/{scene}_vh_clean_2.ply'
    path_mesh_gt_TSDF = path_mesh_gt[:-4]+'_TSDF.ply'
    EvauationUtils.construct_TSDF(path_mesh_gt,
                        path_mesh_gt_TSDF,
                        scene_name=scene,
                        dir_scan=dir_scan,
                        target_img_size=target_img_size,
                        check_existence=True)
    
    # contrust TSDF of reconstructed mesh 
    path_mesh_pred = os.path.join(snerf_scene_exp, 'mesh_reconstruction', f'mesh_canonical_level_{level}_clean.ply')
    path_mesh_pred_TSDF = path_mesh_pred[:-4]+'_TSDF.ply'
    EvauationUtils.construct_TSDF(path_mesh_pred,
                    path_mesh_pred_TSDF,
                    scene_name=scene,
                    dir_scan=dir_scan,
                    target_img_size=target_img_size,
                    check_existence=True)
    
    # compute metrics
    metrices = EvauationUtils.evaluate_geometry_neucon(path_mesh_pred_TSDF, 
                                            path_mesh_gt_TSDF, 
                                            threshold=[0.05], 
                                            down_sample=.02)
    metric_all.append(metrices)

path_log = f'{snerf_exp_root}/eval3Dmesh_{level}.md'
markdown_header=f'| scene_name   |    Method|    Accu.|    Comp.|    Prec.|   Recall|  F-score|  Chamfer\n'
markdown_header=markdown_header+'| -------------| ---------| ------- | ------- | ------- | ------- | ------- | ------- |\n'
EvauationUtils.save_evaluation_results_to_markdown(path_log, 
                                                header = markdown_header, 
                                                name_baseline='s-nerf',
                                                results = metric_all, 
                                                names_item = scene_lis,  
                                                mode = 'w')
    



