# search_top5.py
import open3d as o3d
import numpy as np
import joblib, os
import sys

VOXEL   = 0.003
TRIM_FRAC = 0.30          # 剔除最远 30 % 点，抗残缺
TOPK    = 5

def load_db(db_path):
    db = joblib.load(db_path)
    return db['X'], db['names']

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

def preprocess(stl_file):
    # pcd = o3d.io.read_point_cloud(pcd_path)
    # pcd = pcd.voxel_down_sample(VOXEL)
    # pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL*2, max_nn=30))
    # return pcd
    mesh = o3d.io.read_triangle_mesh(stl_file)
    pcd = mesh.sample_points_uniformly(number_of_points=50000)
    return pcd

def trimmed_chamfer(src, tgt, trim=TRIM_FRAC):
    """双向 trimmed Chamfer 距离"""
    dist1 = src.compute_point_cloud_distance(tgt)
    dist2 = tgt.compute_point_cloud_distance(src)
    # 各取最短 (1-trim) 比例
    k1 = int(len(dist1)*(1-trim))
    k2 = int(len(dist2)*(1-trim))
    return np.mean(np.partition(dist1, k1)[:k1]) + \
           np.mean(np.partition(dist2, k2)[:k2])

def global_match(feat_query, feat_db):
    """FPFH 均值 L2 距离粗排序，返回前 20 候选"""
    dist = np.linalg.norm(feat_db - feat_query, axis=1)
    return np.argsort(dist)[:20]

def fine_match(query, candidate_paths, topk=TOPK):
    scores = []
    for i, stl_file in enumerate(candidate_paths):
        print(f'Fine matching with:{stl_file}   {i} / {len(candidate_paths)}')
        
        # 读取 STL 网格文件并采样点云
        mesh = o3d.io.read_triangle_mesh(stl_file)
        model = mesh.sample_points_uniformly(number_of_points=50000)
        
        # 确保模型点云有法线
        model.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL*2, max_nn=30))
        
        try:
            # 方案1: 跳过配准，直接计算多尺度 Chamfer 距离 (最快)
            # 居中对齐
            query_centered = query.translate(-query.get_center())
            model_centered = model.translate(-model.get_center())
            
            # 尝试多个旋转角度 (简化版配准)
            best_cd = float('inf')
            for angle in [0, 90, 180, 270]:  # 绕Z轴旋转
                R = query_centered.get_rotation_matrix_from_xyz((0, 0, np.radians(angle)))
                query_rot = query_centered.rotate(R, center=(0, 0, 0))
                cd = trimmed_chamfer(query_rot, model_centered)
                if cd < best_cd:
                    best_cd = cd
            
            scores.append((stl_file, best_cd))
            print(f'  -> CD: {best_cd:.4f}')
            
        except Exception as e:
            print(f'Error processing {stl_file}: {e}')
            continue
    
    scores.sort(key=lambda x: x[1])
    return scores[:topk]

# 在 main 中直接返回粗匹配结果
if __name__ == '__main__':
    db_feat, db_names = load_db('model_db.pkl')
    input_ply = sys.argv[1]
    query_pcd = preprocess(input_ply)
    query_feat = compute_fpfh(query_pcd).mean(axis=0)
    print("\nProcessing query_pcd:", query_pcd.points.__len__(), "points")

    coarse_idx = global_match(query_feat, db_feat)
    print("Coarse match candidates:", coarse_idx)
    
    # 直接使用粗匹配结果（最快，跳过精细匹配）
    top5_idx = coarse_idx[:TOPK]
    for idx in top5_idx:
        dist = np.linalg.norm(db_feat[idx] - query_feat)
        print(f'{db_names[idx]}  ->  FPFH-dist = {dist:.4f}')