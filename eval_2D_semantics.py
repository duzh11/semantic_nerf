import os, shutil
import json
import cv2
import numpy as np  
from glob import glob
from tqdm import tqdm

import utils_evaulation as EvauationUtils

scene_lis = ['scene0378_00', 'scene0435_02', 'scene0050_00', 'scene0616_00', \
             'scene0084_00', 'scene0169_00', 'scene0025_00', 'scene0426_00']

neuris_data_root = '/home/du/Proj/3Dv_Reconstruction/NeuRIS/Data/dataset/indoor'
snerf_data_root = '/home/du/Proj/3Dv_Semantics/semantic_nerf/Data'
snerf_exp_root = '/home/du/Proj/3Dv_Semantics/semantic_nerf/exps'

metric_all = []
for scene in tqdm(scene_lis, desc='Evaluating 2D Semantic'):
    neuris_scene_root = os.path.join(neuris_data_root, scene)
    snerf_scene_exp = os.path.join(snerf_exp_root, scene+'_NeuRIS')

    semantic_GT_lis = sorted(glob(os.path.join(neuris_scene_root, 'semantic/train/semantic_GT', '*.png')))
    semantic_render_lis = sorted(glob(os.path.join(snerf_scene_exp, 'train_render/step_200000', 'label_*.png')))

    semantic_GT_list=[]
    semantic_render_list=[]
    for idx in range(len(semantic_GT_lis)):
        semantic_GT = (cv2.imread(semantic_GT_lis[idx], cv2.IMREAD_UNCHANGED)).astype(np.uint8)
        semantic_render = (cv2.imread(semantic_render_lis[idx], cv2.IMREAD_UNCHANGED)).astype(np.uint8)

        # 考虑渲染的语义是(320,240)
        reso=semantic_GT.shape[0]/semantic_render.shape[0]
        if reso>1:
            semantic_GT=cv2.resize(semantic_GT, (semantic_render.shape[1],semantic_render.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        semantic_GT_list.append(np.array(semantic_GT))
        semantic_render_list.append(semantic_render)

    true_labels=np.array(semantic_GT_list)
    predicted_labels=np.array(semantic_render_list)  

    ##### obtain remap #####
    snerf_scene_data = os.path.join(snerf_data_root, scene+'_NeuRIS')
    semantic_data_lis = sorted(glob(os.path.join(snerf_scene_data, f'{scene}_2d-label-filt/label-filt/*.png')))
    semantic_data_list=[]
    for idx in range(len(semantic_data_lis)):
        semantic_data = (cv2.imread(semantic_data_lis[idx], cv2.IMREAD_UNCHANGED)).astype(np.uint8)
        semantic_data_resize = cv2.resize(semantic_data, (semantic_render.shape[1],semantic_render.shape[0]), interpolation=cv2.INTER_NEAREST)
        semantic_data_list.append(np.array(semantic_data))
    data_labels=np.array(semantic_data_list)

    remap_dict = {}
    class_name_string = ["void",
        "wall", "floor", "cabinet", "bed", "chair",
        "sofa", "table", "door", "window", "book", 
        "picture", "counter", "blinds", "desk", "shelves",
        "curtain", "dresser", "pillow", "mirror", "floor",
        "clothes", "ceiling", "books", "fridge", "tv",
        "paper", "towel", "shower curtain", "box", "white board",
        "person", "night stand", "toilet", "sink", "lamp",
        "bath tub", "bag", "other struct", "other furntr", "other prop"] # NYUv2-40-class
    
    data_classes = np.unique(data_labels)
    for idx in range(len(data_classes)):
        # remap_idx: [original_idx, class_name]
        remap_dict[idx] = [int(data_classes[idx]), class_name_string[data_classes[idx]]]
    
    with open(os.path.join(snerf_scene_exp, 'remap.json'), 'w') as f:
        json.dump(remap_dict, f)

    # remap predicted labels
    predicted_labels_remapped = predicted_labels.copy()
    for label_idx in np.unique(predicted_labels):
        predicted_labels_remapped[predicted_labels==label_idx] = remap_dict[label_idx][0]
    
    # compute metrics
    true_labels = true_labels-1
    predicted_labels_remapped = (predicted_labels_remapped-1).astype(np.uint8)
    metric_avg, exsiting_label, class_iou, class_accuray = EvauationUtils.compute_segmentation_metrics(true_labels=true_labels, 
                                                                                                       predicted_labels=predicted_labels_remapped, 
                                                                                                       ignore_label=255)
    metric_all.append(metric_avg)

path_log = f'{snerf_exp_root}/eval2Dsemantic.md'
markdown_header='train\n| scene_ name   |   Method|  Acc|  M_Acc|  M_IoU| FW_IoU|\n'
markdown_header=markdown_header+'| -------------| ---------| ----- | ----- | ----- | ----- |\n'
EvauationUtils.save_evaluation_results_to_markdown(path_log, 
                                                header = markdown_header, 
                                                name_baseline='s-nerf',
                                                results = metric_all, 
                                                names_item = scene_lis,  
                                                mode = 'w')
    



