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

def build_db(model_dir, save_path):
    feats, names = [], []
    for stl_file in glob.glob(os.path.join(model_dir, "*.stl")):
        # 读取 STL 网格文件
        mesh = o3d.io.read_triangle_mesh(stl_file)
        
        print("Mesh has", len(mesh.vertices))
        
        # 从网格采样点云
        pcd = mesh.sample_points_uniformly(number_of_points=10000)
        
        try:
            feat = compute_fpfh(pcd).mean(axis=0)   # 全局 33 维 FPFH 均值
            feats.append(feat)
            names.append(os.path.basename(stl_file))
            print(f'Processed: {os.path.basename(stl_file)}')
        except Exception as e:
            print(f'Error processing {stl_file}: {e}')
            continue
    
    if len(feats) == 0:
        raise ValueError("No valid features extracted from any model")
    
    joblib.dump({'X': np.vstack(feats), 'names': names}, save_path)
    print('DB built:', save_path)

if __name__ == '__main__':
    build_db('model_library', 'model_db.pkl')