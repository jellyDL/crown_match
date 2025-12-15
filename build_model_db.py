"建立数据库"
# build_model_db.py
import open3d as o3d
import numpy as np
import joblib, os, glob

def compute_fpfh(pcd, voxel=0.003):
    # 检查点云是否为空
    if len(pcd.points) == 0:
        raise ValueError("Point cloud is empty")
    
    pcd = pcd.voxel_down_sample(voxel)
    
    # 检查降采样后是否还有点
    if len(pcd.points) == 0:
        raise ValueError("Point cloud is empty after downsampling")
    
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2, max_nn=30))
    
    # 检查法线是否计算成功
    if not pcd.has_normals():
        raise ValueError("Failed to compute normals")
    
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*5, max_nn=100))
    return np.asarray(fpfh.data).T      # (N, 33)

def compute_multi_scale_fpfh(pcd):
    """计算多尺度 FPFH 特征"""
    features = []
    for voxel in [0.002, 0.003, 0.005]:
        pcd_down = pcd.voxel_down_sample(voxel)
        if len(pcd_down.points) < 10:
            continue
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2, max_nn=30))
        if not pcd_down.has_normals():
            continue
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*5, max_nn=100))
        feat = np.asarray(fpfh.data).T.mean(axis=0)
        features.append(feat)
    
    if len(features) == 0:
        raise ValueError("Failed to compute FPFH features")
    
    return np.concatenate(features)

def compute_shape_descriptor(pcd):
    """计算形状描述符"""
    points = np.asarray(pcd.points)
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    
    max_extent = np.max(extent)
    if max_extent > 0:
        ratio = extent / max_extent
    else:
        ratio = np.ones(3)
    
    volume = np.prod(extent) if np.prod(extent) > 0 else 1.0
    density = len(points) / volume
    center = pcd.get_center()
    center_offset = np.linalg.norm(center - bbox.get_center())
    
    return np.array([*ratio, np.log1p(density), center_offset])

def build_db(model_dir, save_path):
    feats, names = [], []
    for stl_file in glob.glob(os.path.join(model_dir, "*.stl")):
        # 读取 STL 网格文件
        mesh = o3d.io.read_triangle_mesh(stl_file)
        
        print("Mesh has", len(mesh.vertices), "vertices")
        
        # 从网格采样点云
        pcd = mesh.sample_points_uniformly(number_of_points=10000)
        
        # 截取顶部1/3
        points = np.asarray(pcd.points)
        z_coords = points[:, 2]
        # 保留比例
        # ratio = 1/3
        ratio = 1
        z_threshold = np.percentile(z_coords, (1 - ratio) * 100)
        top_indices = z_coords >= z_threshold
        pcd = pcd.select_by_index(np.where(top_indices)[0])
        
        try:
            feat_fpfh = compute_multi_scale_fpfh(pcd)
            feat_shape = compute_shape_descriptor(pcd)
            feat = np.concatenate([feat_fpfh, feat_shape])
            
            feats.append(feat)
            names.append(os.path.basename(stl_file))
            print(f'Processed: {os.path.basename(stl_file)} (feat dim: {len(feat)})')
        except Exception as e:
            print(f'Error processing {stl_file}: {e}')
            continue
    
    if len(feats) == 0:
        raise ValueError("No valid features extracted from any model")
    
    joblib.dump({'X': np.vstack(feats), 'names': names}, save_path)
    print('DB built:', save_path)

if __name__ == '__main__':
    build_db('model_library', 'model_db.pkl')