import os, shutil

scene_lis = ['scene0050_00', 'scene0616_00', 'scene0169_00', 'scene0426_00', \
             'scene0084_00', 'scene0025_00', ]
neuris_data_root = '/home/du/Proj/3Dv_Reconstruction/NeuRIS/Data/dataset/indoor'
snerf_data_root = '/home/du/Proj/3Dv_Semantics/semantic_nerf/Data'

for scene in scene_lis:
    neuris_scene_root = os.path.join(neuris_data_root, scene)
    snerf_scene_root = os.path.join(snerf_data_root, scene+'_NeuRIS')

    # copy predicted semantcis
    file_num_lis = []
    for data_mode in ['train', 'test']:
        neuris_scene_semantic_dir = os.path.join(os.path.join(neuris_scene_root, 'semantic', data_mode, 'deeplab'))
        snerf_scene_semantic_dir = os.path.join(snerf_scene_root, f'{scene}_2d-label-filt/label-filt')
        os.makedirs(snerf_scene_semantic_dir, exist_ok=True)
        
        semantic_file_lis = os.listdir(neuris_scene_semantic_dir)
        file_mode_lis = [int(os.path.splitext(semantic_file_idx)[0]) for semantic_file_idx in semantic_file_lis]
        file_num_lis.extend(file_mode_lis)

        for semantic_file_id in semantic_file_lis:
            source_file = os.path.join(neuris_scene_semantic_dir, semantic_file_id)
            target_file = os.path.join(snerf_scene_semantic_dir, 
                                       str(int(os.path.splitext(semantic_file_id)[0]))+'.png')
            print(f'copying {source_file} to {target_file}')
            shutil.copy(source_file, target_file)

    file_num_lis.sort()
    data_dir_lis = ['color', 'depth', 'pose', f'{scene}_2d-instance-filt/instance-filt']
    for data_dir in data_dir_lis:
        snerf_data_dir = os.path.join(snerf_scene_root, data_dir)
        file_lis = os.listdir(snerf_data_dir)
        #  if file_idx % 10 ==0 then delte
        for file in file_lis:
            if int(os.path.splitext(file)[0]) not in file_num_lis:
                os.remove(os.path.join(snerf_data_dir, file))
    

            