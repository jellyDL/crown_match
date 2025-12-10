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
    pcd = mesh.sample_points_uniformly(number_of_points=1000)
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
        model = mesh.sample_points_uniformly(number_of_points=1000)
        
        # 确保模型点云有法线
        model.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL*2, max_nn=30))
        
        try:
            # 1. 粗配准：ISS + RANSAC (优化参数以提速)
            trans = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                query, model,
                o3d.pipelines.registration.compute_fpfh_feature(
                    query, o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL*5, max_nn=100)),
                o3d.pipelines.registration.compute_fpfh_feature(
                    model, o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL*5, max_nn=100)),
                mutual_filter=True,  # 启用互滤波加速
                max_correspondence_distance=VOXEL*20,  # 增大距离阈值
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=3,  # 减少采样点数
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000)  # 大幅减少迭代次数
            ).transformation
            
            query_t = query.transform(trans)
            
            # 2. 精配准 + trimmed CD
            icp = o3d.pipelines.registration.registration_icp(
                query_t, model, VOXEL*2, trans,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))  # 减少ICP迭代
            query_final = query.transform(icp.transformation)
            cd = trimmed_chamfer(query_final, model)
            scores.append((stl_file, cd))
            print(f'  -> CD: {cd:.4f}')
        except Exception as e:
            print(f'Error processing {stl_file}: {e}')
            continue
    
    scores.sort(key=lambda x: x[1])
    return scores[:topk]

if __name__ == '__main__':
    db_feat, db_names = load_db('model_db.pkl')
    input_ply = sys.argv[1]
    query_pcd = preprocess(input_ply)
    query_feat = compute_fpfh(query_pcd).mean(axis=0)
    print("Processing query_pcd:", query_pcd.points.__len__(), "points")

    coarse_idx = global_match(query_feat, db_feat)
    print("Coarse match candidates:", coarse_idx)
    candidate_paths = [os.path.join('model_library', db_names[i]) for i in coarse_idx]
    # print("Coarse match candidate_paths:", candidate_paths)

    top5 = fine_match(query_pcd, candidate_paths)
    
    for ply, score in top5:
        print(f'{ply}  ->  trimmed-CD = {score:.4f}')