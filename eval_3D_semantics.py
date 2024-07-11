import os, shutil
import json
import cv2
import numpy as np  
from glob import glob
from tqdm import tqdm

import utils_evaulation as EvauationUtils

level, neart, fart=0.45, 5.0, 10.0
scene_lis = ['scene0378_00', 'scene0435_02', 'scene0050_00', 'scene0616_00', \
             'scene0084_00', 'scene0169_00', 'scene0025_00', 'scene0426_00']

neuris_data_root = '/home/du/Proj/3Dv_Reconstruction/NeuRIS/Data/dataset/indoor'
snerf_data_root = '/home/du/Proj/3Dv_Semantics/semantic_nerf/Data'
snerf_exp_root = '/home/du/Proj/3Dv_Semantics/semantic_nerf/exps'

metric_surface_all, metric_volume_all = [], []
for scene in tqdm(scene_lis, desc='Evaluating 2D Semantic'):
    neuris_scene_root = os.path.join(neuris_data_root, scene)
    snerf_scene_exp = os.path.join(snerf_exp_root, scene+'_NeuRIS')

    # remap
    with open(os.path.join(snerf_scene_exp, 'remap.json'), 'r') as f:
        remap_dict = json.load(f)
    
    # volume
    file_mesh_trgt =f'{neuris_scene_root}/{scene}_vh_clean_2.labels.ply'
    file_mesh_pred = os.path.join(snerf_scene_exp, 'mesh_reconstruction/use_vertex_normal', f'semantic_mesh_canonical_level_{level}_dim_256_neart_{neart}_fart_{fart}.ply')
    file_semseg_pred = os.path.join(snerf_scene_exp, 'mesh_reconstruction/use_vertex_normal', f'semantic_mesh_canonical_level_{level}_label_dim_256_neart_{neart}_fart_{fart}.npz')
    file_mesh_pred_transfer = file_mesh_pred[:-4]+'_transfer.ply'

    mesh_transfer, metric_avg, exsiting_label, class_iou, class_accuray = EvauationUtils.evaluate_semantic_3D(file_mesh_trgt,
                                                                                                                file_mesh_pred,
                                                                                                                file_semseg_pred,
                                                                                                                remap_dict)
    mesh_transfer.export(file_mesh_pred_transfer)
    metric_volume_all.append(metric_avg)

    # surface
    file_mesh_trgt =f'{neuris_scene_root}/{scene}_vh_clean_2.labels.ply'
    file_mesh_pred = os.path.join(snerf_scene_exp, 'mesh_reconstruction/use_vertex_normal', f'semantic_mesh_canonical_level_{level}_surface_dim_256.ply')
    file_semseg_pred = os.path.join(snerf_scene_exp, 'mesh_reconstruction/use_vertex_normal', f'semantic_mesh_canonical_level_{level}_label_surface_dim_256.npz')
    file_mesh_pred_transfer = file_mesh_pred[:-4]+'_transfer.ply'

    mesh_transfer, metric_avg, exsiting_label, class_iou, class_accuray = EvauationUtils.evaluate_semantic_3D(file_mesh_trgt,
                                                                                                                file_mesh_pred,
                                                                                                                file_semseg_pred,
                                                                                                                remap_dict)
    mesh_transfer.export(file_mesh_pred_transfer)
    metric_surface_all.append(metric_avg)


path_log = f'{snerf_exp_root}/eval3Dsemantic_level_{level}_neart_{neart}_fart_{fart}.md'
markdown_header='volume\n| scene_ name   |   Method|  Acc|  M_Acc|  M_IoU| FW_IoU|\n'
markdown_header=markdown_header+'| -------------| ---------| ----- | ----- | ----- | ----- |\n'
EvauationUtils.save_evaluation_results_to_markdown(path_log, 
                                                header = markdown_header, 
                                                name_baseline='s-nerf',
                                                results = metric_volume_all, 
                                                names_item = scene_lis,  
                                                mode = 'w')

markdown_header='\nsurface\n| scene_ name   |   Method|  Acc|  M_Acc|  M_IoU| FW_IoU|\n'
markdown_header=markdown_header+'| -------------| ---------| ----- | ----- | ----- | ----- |\n'
EvauationUtils.save_evaluation_results_to_markdown(path_log, 
                                                header = markdown_header, 
                                                name_baseline='s-nerf',
                                                results = metric_surface_all, 
                                                names_item = scene_lis,  
                                                mode = 'a')



