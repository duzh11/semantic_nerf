from sklearn.metrics import confusion_matrix
from glob import glob
from tqdm import tqdm
import numpy as np
import open3d as o3d

import trimesh
import pyrender
import os
import cv2

nyu40_colour_code = np.array([
       (0, 0, 0),

       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair

       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf

       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),       # blinds
       (247, 182, 210),		# desk
       (66, 188, 102),      # shelves

       (219, 219, 141),		# curtain
       (140, 57, 197),    # dresser
       (202, 185, 52),      # pillow
       (51, 176, 203),    # mirror
       (200, 54, 131),      # floor

       (92, 193, 61),       # clothes
       (78, 71, 183),       # ceiling
       (172, 114, 82),      # books
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),      # tv

       (153, 98, 156),      # paper
       (140, 153, 101),     # towel
       (158, 218, 229),		# shower curtain
       (100, 125, 154),     # box
       (178, 127, 135),       # white board

       (120, 185, 128),       # person
       (146, 111, 194),     # night stand
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),      # lamp

       (227, 119, 194),		# bathtub
       (213, 92, 176),      # bag
       (94, 106, 211),      # other struct
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)       # other prop
    ]).astype(np.uint8)


def save_evaluation_results_to_markdown(path_log, 
                                        header = '                     Accu.      Comp.      Prec.     Recall     F-score \n', 
                                        name_baseline=None,
                                        results = None, 
                                        names_item = None, 
                                        save_mean = True, 
                                        mode = 'w',
                                        precision = 3):
    '''Save evaluation results to txt in latex mode
    Args:
        header:
            for F-score: '                     Accu.      Comp.      Prec.     Recall     F-score \n'
        results:
            narray, N*M, N lines with M metrics
        names_item:
            N*1, item name for each line
        save_mean: 
            whether calculate the mean value for each metric
        mode:
            write mode, default 'w'
    '''
    # save evaluation results to latex format
    results=np.array(results)
    with open(path_log, mode) as f_log:
        if header:
            f_log.writelines(header)
        if results is not None:
            num_lines, num_metrices = results.shape
            if names_item is None:
                names_item = np.arange(results.shape[0])
            for idx in range(num_lines):
                f_log.writelines((f'|{names_item[idx]}  | {name_baseline}|' + ("{: 8.3f}|" * num_metrices).format(*results[idx, :].tolist())) + " \n")
        if save_mean:
            mean_results = np.nanmean(results,axis=0)     # 4*7
            mean_results = np.round(mean_results, decimals=precision)
            f_log.writelines(( f'|       Mean  | {name_baseline}|' + "{: 8.3f}|" * num_metrices).format(*mean_results[:].tolist()) + " \n")

def compute_segmentation_metrics(true_labels, 
                                 predicted_labels, 
                                 semantic_class=40, 
                                 ignore_label=-1):

    true_labels=np.array(true_labels)
    predicted_labels=np.array(predicted_labels)
    
    if (true_labels == ignore_label).all():
        return [0]*4

    true_labels = true_labels.flatten()
    predicted_labels = predicted_labels.flatten()
    valid_pix_ids = true_labels!=ignore_label
    predicted_labels = predicted_labels[valid_pix_ids] 
    true_labels = true_labels[valid_pix_ids]
    
    # 利用confusion matrix进行计算
    conf_mat = confusion_matrix(true_labels, predicted_labels, labels=list(range(0, semantic_class)))
    norm_conf_mat = np.transpose(
        np.transpose(conf_mat) / conf_mat.astype(float).sum(axis=1))

    missing_class_mask = np.isnan(norm_conf_mat.sum(1)) # missing class will have NaN at corresponding class
    exsiting_class_mask = ~ missing_class_mask

    if semantic_class==3:
        label=np.array(["object", "wall", "floor"])
    elif semantic_class==40:
        label = np.array(["wall", "floor", "cabinet", "bed", "chair",
                "sofa", "table", "door", "window", "book", 
                "picture", "counter", "blinds", "desk", "shelves",
                "curtain", "dresser", "pillow", "mirror", "floor",
                "clothes", "ceiling", "books", "fridge", "tv",
                "paper", "towel", "shower curtain", "box", "white board",
                "person", "night stand", "toilet", "sink", "lamp",
                "bath tub", "bag", "other struct", "other furntr", "other prop"])
        
    exsiting_label = label[exsiting_class_mask]
    # ACC
    average_accuracy = np.mean(np.diagonal(norm_conf_mat)[exsiting_class_mask]) #平均精度
    total_accuracy = (np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat)) #总精度
    class_accuray_0=np.diagonal(norm_conf_mat).copy() #类别精度
    class_accuray=class_accuray_0[exsiting_class_mask]
    
    # IoU
    class_iou_0 = np.zeros(semantic_class)
    for class_id in range(semantic_class):
        class_iou_0[class_id] = (conf_mat[class_id, class_id] / (
                np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) -
                conf_mat[class_id, class_id])) 
    
    class_iou = class_iou_0[exsiting_class_mask]
    average_iou = np.mean(class_iou) #平均IoU
    freq = conf_mat.sum(axis=1) / conf_mat.sum()
    FW_iou = (freq[exsiting_class_mask] * class_iou_0[exsiting_class_mask]).sum()

    metric_avg = [average_accuracy, total_accuracy, average_iou, FW_iou]
    return metric_avg, exsiting_label, class_iou, class_accuray

class Renderer():
    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        # self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)  # , self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()

def refuse(mesh, intrinsic_depth, rgb_all, c2w_all, acc=0.01):
    #将颜色和几何点云混合？
    renderer = Renderer()
    mesh_opengl = renderer.mesh_opengl(mesh)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=acc,
        sdf_trunc=3 * acc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    n_image=rgb_all.shape[0]

    for i in tqdm(range(n_image), desc='refusing'):
        h, w = rgb_all[i].shape[0], rgb_all[i].shape[1]

        intrinsic = np.eye(4)
        intrinsic[:3, :3] = intrinsic_depth
        pose = c2w_all[i]
        pose[:3, 3]=pose[:3, 3]
        rgb = rgb_all[i]
        rgb = (rgb * 255).astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)
        _, depth_pred = renderer(h, w, intrinsic, pose, mesh_opengl)
        depth_pred = o3d.geometry.Image(depth_pred)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth_pred, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=w, height=h, fx=fx,  fy=fy, cx=cx, cy=cy)
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, intrinsic, extrinsic)
    
    return volume.extract_triangle_mesh()

def construct_TSDF(path_mesh,
                    path_mesh_TSDF,
                    scene_name,
                    dir_scan,
                    target_img_size,
                    check_existence=True):
    
    image_dir=os.path.join(dir_scan, 'image/train', '*.png')
    image_list=sorted(glob(image_dir))

    c2w_all=[]
    rgb_all=[]
    
    if check_existence and os.path.exists(path_mesh_TSDF):
        print(f'The mesh TSDF has been constructed. [{path_mesh_TSDF.split("/")[-1]}]')
        return -1

    for img_name in image_list:
        c2w = np.loadtxt(f'{dir_scan}/pose/train/{img_name[-8:-4]}.txt')
        c2w_all.append(c2w)

        rgb = cv2.imread(img_name)
        reso=target_img_size[0]/rgb.shape[0]
        
        if reso>2:
            rgb=rgb.astype(np.uint8)
            rgb=cv2.pyrUp(rgb)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = (rgb.astype(np.float32) / 255)

        rgb_all.append(rgb)
    
    intrinsic_dir=f'{dir_scan}/intrinsic_depth.txt'
    intrinsic_depth=np.loadtxt(intrinsic_dir)
    
    print(f'construct TSDF: {scene_name}')
    mesh_pred=trimesh.load(path_mesh)
    mesh_pred_REFUSE = refuse(mesh_pred, intrinsic_depth[:3, :3], np.array(rgb_all), np.array(c2w_all))
    o3d.io.write_triangle_mesh(path_mesh_TSDF, mesh_pred_REFUSE)

def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1

    Args:
        nx3 np.array's

    Returns:
        ([indices], [distances])

    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances

def evaluate_geometry_neucon(file_pred, 
                             file_trgt, 
                             threshold=.05, 
                             down_sample=.02):
    """ Borrowed from NeuralRecon
    Compute Mesh metrics between prediction and target.

    Opens the Meshs and runs the metrics

    Args:
        file_pred: file path of prediction
        file_trgt: file path of target
        threshold: distance threshold used to compute precision/recal
        down_sample: use voxel_downsample to uniformly sample mesh points

    Returns:
        Dict of mesh metrics
    """

    pcd_pred = o3d.io.read_point_cloud(file_pred)
    pcd_trgt = o3d.io.read_point_cloud(file_trgt)
    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)
    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    ind1, dist1 = nn_correspondance(verts_pred, verts_trgt)  # para2->para1: dist1 is gt->pred
    ind2, dist2 = nn_correspondance(verts_trgt, verts_pred)
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    chamfer= np.mean(dist1**2)+np.mean(dist2**2)
    metrics = {'dist1': np.mean(dist2),  # pred->gt
               'dist2': np.mean(dist1),  # gt -> pred
               'prec': precision,
               'recal': recal,
               'fscore': fscore,
               'chamfer': chamfer,
               }

    metrics = np.array([np.mean(dist2), np.mean(dist1), precision, recal, fscore, chamfer])
    print(f'{file_pred.split("/")[-1]}: {metrics}')
    
    return metrics

def project_to_mesh(from_mesh, to_mesh, attribute, dist_thresh=None, cmap=None):
    """ Transfers attributs from from_mesh to to_mesh using nearest neighbors

    Each vertex in to_mesh gets assigned the attribute of the nearest
    vertex in from mesh. Used for semantic evaluation.

    Args:
        from_mesh: Trimesh with known attributes
        to_mesh: Trimesh to be labeled
        attribute: Which attribute to transfer
        dist_thresh: Do not transfer attributes beyond this distance
            (None transfers regardless of distacne between from and to vertices)

    Returns:
        Trimesh containing transfered attribute
    """

    if len(from_mesh.vertices) == 0:
        to_mesh.vertex_attributes[attribute] = np.zeros((0), dtype=np.uint8)
        to_mesh.visual.vertex_colors = np.zeros((0), dtype=np.uint8)
        return to_mesh

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(from_mesh.vertices)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    pred_ids = from_mesh.vertex_attributes[attribute]
    pred_ids[pred_ids == 255] = 0
    
    matched_ids = np.zeros((to_mesh.vertices.shape[0]), dtype=np.uint8)
    matched_colors = np.ones((to_mesh.vertices.shape[0], 4), dtype=np.uint8) * 255
    
    for i, vert in enumerate(to_mesh.vertices):
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        if dist_thresh is None or dist[0]<dist_thresh:
            matched_ids[i] = pred_ids[inds[0]]
            matched_colors[i][:3] = cmap[int(pred_ids[inds[0]])]

    mesh = to_mesh.copy()
    mesh.vertex_attributes['label'] = matched_ids
    mesh.visual.vertex_colors = matched_colors
    return mesh

def read_label(fn, is_gt=False):
    import plyfile
    a = plyfile.PlyData().read(fn)
    w = np.array(a.elements[0]['label'])
    
    w = w.astype(np.uint8)
    return w

def evaluate_semantic_3D(file_mesh_trgt,
                         file_mesh_pred,
                         file_semseg_pred,
                         remap_dict,
                         semantic_class=40,
                         MANHATTAN=False):
    mesh_trgt = trimesh.load(file_mesh_trgt, process=False)
    mesh_pred = trimesh.load(file_mesh_pred, process=False)
    semseg_pred = np.load(file_semseg_pred)['arr_0']

    # remap semseg labels
    semseg_pred_remapped = semseg_pred.copy()
    for label_idx in np.unique(semseg_pred):
        semseg_pred_remapped[semseg_pred==label_idx] = remap_dict[str(label_idx-1)][0]

    vertex_attributes = {}
    vertex_attributes['semseg'] = semseg_pred_remapped.astype(np.uint8)
    mesh_pred.vertex_attributes = vertex_attributes

    # transfer labels from pred mesh to gt mesh using nearest neighbors
    colour_map_np = nyu40_colour_code
    mesh_transfer = project_to_mesh(mesh_pred, mesh_trgt, 'semseg', cmap=colour_map_np)
    semseg_pred_trasnfer = mesh_transfer.vertex_attributes['label']

    pred_ids = semseg_pred_trasnfer
    gt_ids = read_label(file_mesh_trgt)

    # merge语义
    semantic_GT = np.array(gt_ids)
    semantic_render = np.array(pred_ids)
    semantic_GT_copy, semantic_render_copy = semantic_GT.copy(), semantic_render.copy()

    if semantic_class>3:
        true_labels=semantic_GT_copy-1
        predicted_labels=semantic_render_copy-1
    else:
        true_labels=semantic_GT_copy
        predicted_labels=semantic_render_copy  

    metric_avg, exsiting_label, class_iou, class_accuray = compute_segmentation_metrics(true_labels=true_labels, 
                                                                                        predicted_labels=predicted_labels, 
                                                                                        semantic_class=semantic_class, 
                                                                                        ignore_label=255)
    
    return mesh_transfer, metric_avg, exsiting_label, class_iou, class_accuray