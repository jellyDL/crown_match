# search_top5.py
import open3d as o3d
import numpy as np
import joblib, os
import sys

VOXEL   = 0.003
TRIM_FRAC = 0.30          # 剔除最远 30 % 点，抗残缺
TOPK    = 3

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

def preprocess(stl_file, top_fraction=1/3):
    mesh = o3d.io.read_triangle_mesh(stl_file)
    pcd = mesh.sample_points_uniformly(number_of_points=10000)
    
    # 截取顶部 top_fraction 的点云
    points = np.asarray(pcd.points)
    z_coords = points[:, 2]  # Z轴坐标
    z_threshold = np.percentile(z_coords, (1 - top_fraction) * 100)
    
    # 保留 Z 坐标大于阈值的点
    top_indices = z_coords >= z_threshold
    pcd_top = pcd.select_by_index(np.where(top_indices)[0])
    
    print(f"Original points: {len(points)}, Top {top_fraction*100:.0f}% points: {len(pcd_top.points)}")
    
    return pcd_top

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
        model = mesh.sample_points_uniformly(number_of_points=10000)
        
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

def match_single_file(input_ply):
    
    # 只截取顶部1/3的点云进行匹配
    query_pcd = preprocess(input_ply, top_fraction=2/3)
    o3d.io.write_point_cloud("query_pcd.ply", query_pcd)
    
    query_feat = compute_fpfh(query_pcd).mean(axis=0)
    print("\nProcessing query_pcd:", query_pcd.points.__len__(), "points")

    coarse_idx = global_match(query_feat, db_feat)
    # print("Coarse match candidates:", coarse_idx, "\n")
    
    # 直接使用粗匹配结果（最快，跳过精细匹配）
    top5_idx = coarse_idx[:TOPK]
    for i, idx in enumerate(top5_idx):
        dist = np.linalg.norm(db_feat[idx] - query_feat)
        if i == 0:
            print("[*] ", end="")
        print(f'{db_names[idx]:<60}  ->  FPFH-dist = {dist:.4f}')

    if db_names[top5_idx[0]] == input_ply.split('/')[-1]:
        print("\n======> Top-1 match is correct!")
        return True
    else:
        print("\n======> Top-1 match is incorrect!")
        return False
        

# 在 main 中直接返回粗匹配结果
if __name__ == '__main__':
    db_feat, db_names = load_db('model_db.pkl')
    
    if len(sys.argv) < 2:
        print("Usage: python match.py <ply or folder>")
        exit(1)
        
    input_param = sys.argv[1]
    if os.path.isfile(input_param):    
        match_single_file(input_param)
        
    elif os.path.isdir(input_param):
        folder = input_param
        total_count = 0
        success_count = 0
        for file in os.listdir(folder):
            print("Checking file:", file)
            if file.endswith('.stl'):
                input_ply = os.path.join(folder, file)
                print("\n=== Matching for file:", input_ply)
                ret = match_single_file(input_ply)  
                if ret is True:
                    success_count += 1  
                total_count += 1
        print(f"\nOverall Top-1 Accuracy: {success_count} / {total_count} = {success_count/total_count:.2%}")