import os
import torch
import numpy as np
import math
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 【关键引入】：从 v2 文件导入
from hololens_head_pose_reprojection_v2 import (
    get_device, create_camera, create_raster_settings, 
    create_rasterizer, create_phong_renderer, load_image_as_tensor,
    image_cropped, load_head_mesh, predict_head_pose_t_geometric, 
    predict_head_pose_r_geometric, final_angles, AdaptiveLRScheduler
)
from pytorch3d.transforms import euler_angles_to_matrix
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles
from tqdm import tqdm

def p3d_to_opencv_pose(final_t, final_angles_deg):
    r_rad = torch.tensor(final_angles_deg, dtype=torch.float32) * (math.pi / 180.0)
    R_obj = euler_angles_to_matrix(r_rad, "ZXY").numpy()
    T_obj = np.array(final_t, dtype=np.float32)
    
    R_p3d_view = R_obj
    T_p3d_view = T_obj + np.array([0.0, 0.0, 0.2], dtype=np.float32)
    
    S_flip = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]], dtype=np.float32)
    R_cv = S_flip @ R_p3d_view.T
    T_cv = S_flip @ T_p3d_view
    
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R_cv
    pose[:3, 3] = T_cv
    return pose

def save_overlay_image(img_ref, img_render, save_path):
    if img_ref is None or img_render is None: 
        return
    img_ref_np = img_ref[0, ..., :3].cpu().detach().numpy()
    img_render_np = img_render[0, ..., :3].cpu().detach().numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_ref_np)
    plt.title("Input RGB Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_ref_np)
    plt.imshow(img_render_np, alpha=0.6) 
    plt.title("Optimization Overlay Result")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close() 

def main():
    parser = argparse.ArgumentParser(description="Estimate Head Pose from a single RGB image.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the input RGB image (e.g., ./test.png)")
    parser.add_argument("--mesh_dir", type=str, default="./dataset/head", help="Directory containing the head mesh")
    parser.add_argument("--mesh_name", type=str, default="head_col.obj", help="Name of the head mesh file")
    parser.add_argument("--out_dir", type=str, default="./output", help="Directory to save the pose txt and visualization")
    args = parser.parse_args()

    # 检查输入图像是否存在
    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"Input image not found: {args.img_path}")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.img_path))[0]

    # 初始化 Pytorch3D 环境
    device = get_device()
    cameras = create_camera(device)
    raster_settings = create_raster_settings()
    rasterizer = create_rasterizer(device, cameras, raster_settings)
    phong_renderer = create_phong_renderer(device, cameras, raster_settings)
    
    # 载入 Mesh
    head_mesh = load_head_mesh(args.mesh_dir, args.mesh_name, device)
    
    print(f"========== 开始处理单张图像: {args.img_path} ==========")
    
    # 读取图像并裁剪
    img_ref_tensor = load_image_as_tensor(args.img_path, device, (976, 549))
    img_ref_tensor = image_cropped(img_ref_tensor)
    
    # 初始化优化器及参数
    scheduler = AdaptiveLRScheduler(base_lr_t=0.005, base_lr_r=0.005, patience=5, min_delta=1e-5, max_iterations=50)
    final_t = [0.0, 0.0, 0.0]
    final_angles_deg = np.array([0, 0, 0])
    max_outer_iterations = 50 
    early_stop_count = 0
    img_r_result = None 
    
    print("\n========== 开始优化迭代 ==========")
    # 使用 tqdm 包装进度条
    with tqdm(total=max_outer_iterations, desc="Optimizing Pose", unit="iter") as pbar:
        for outer_iter in range(max_outer_iterations):
            head_mesh_t = head_mesh.clone()
            head_mesh_r = head_mesh.clone()
            
            # 优化平移 T
            t_result, _, _, t_conv_info, t_early = predict_head_pose_t_geometric(
                final_t, final_angles_deg, head_mesh_t, rasterizer, phong_renderer, cameras, img_ref_tensor, scheduler, device=device)
            final_t = t_result
            
            # 优化旋转 R
            r_result, _, img_r_result, r_loss, r_conv_info, r_early = predict_head_pose_r_geometric(
                final_t, final_angles_deg, head_mesh_r, rasterizer, phong_renderer, cameras, img_ref_tensor, scheduler, device=device)
            
            # 获取当前 loss 用于进度条显示
            t_loss_val = t_conv_info['final_loss'] if t_conv_info['final_loss'] is not None else float('inf')
            r_loss_val = r_conv_info['final_loss'] if r_conv_info['final_loss'] is not None else float('inf')
            
            # 更新进度条显示的后缀信息
            pbar.set_postfix({
                "T_loss": f"{t_loss_val:.5f}",
                "R_loss": f"{r_loss_val:.5f}"
            })
            pbar.update(1)
            
            # 判断早停
            if t_early and r_early: 
                early_stop_count += 1
            else: 
                early_stop_count = 0
            
            if early_stop_count > 4:
                print("\n[INFO] 触发早停机制，停止优化。")
                break
                
            final_angles_deg = final_angles(final_angles_deg, r_result)
            if isinstance(final_angles_deg, torch.Tensor): 
                final_angles_deg = final_angles_deg.detach().cpu().numpy().tolist()
                
            # 判断收敛
            if t_conv_info['converged'] and r_conv_info['converged']:
                if t_loss_val < 0.001 and r_loss_val < 0.01:
                    print("\n[INFO] 优化已收敛，停止迭代。")
                    break
                
            del head_mesh_t, head_mesh_r
            torch.cuda.empty_cache()
        
        # 计算最终相机位姿 (4x4 矩阵)
    pose_4x4 = p3d_to_opencv_pose(final_t, final_angles_deg)
    
    # 提取 T (x, y, z)
    t_xyz = pose_4x4[:3, 3]
    
    # 提取 R 并转换为欧拉角 (rx, ry, rz)
    R_cv = torch.tensor(pose_4x4[:3, :3], dtype=torch.float32)
    euler_rad = matrix_to_euler_angles(R_cv, "XYZ")
    euler_deg = euler_rad.numpy() * (180.0 / math.pi)
    
    # ========== [新增] 坐标系/欧拉角对齐修正 ==========
    # 根据实际结果对比，手动纠正 Y平移, X旋转, Z旋转 的翻转与 180度偏移
    x, y, z = t_xyz
    rx, ry, rz = euler_deg
    
    # Y轴平移取反
    y_corrected = -y
    
    # X轴旋转取反
    rx_corrected = -rx
    
    # Z轴旋转补偿 180度 后取反 ( -164.8271 -> +180 -> 15.1729 -> 取反 -> -15.1729 )
    # 使用规范化让角度保持在 -180 到 180 之间
    rz_corrected = -(rz + 180.0)
    while rz_corrected > 180.0: rz_corrected -= 360.0
    while rz_corrected < -180.0: rz_corrected += 360.0
    
    # 重新组装为符合实际系统的 [x, y, z, rx, ry, rz]
    final_pose_6dof = np.array([x, y_corrected, z, rx_corrected, ry, rz_corrected])
    # ==================================================
    
    # 打印最终位姿结果
    print("\n========== 优化完成 ==========")
    print("最终预测 6DoF 位姿 [x, y, z, rx, ry, rz] (已修正至实际坐标系):")
    print(np.array_str(final_pose_6dof, precision=4, suppress_small=True))
    
    # 保存 6DoF 位姿 TXT 文件
    pose_txt_path = os.path.join(args.out_dir, f"{stem}_pose.txt")
    np.savetxt(pose_txt_path, final_pose_6dof, header="x y z rx(deg) ry(deg) rz(deg)", comments='')
    print(f"\n[INFO] 位姿已保存至: {pose_txt_path}")
    
    # 保存可视化结果
    vis_path = os.path.join(args.out_dir, f"{stem}_vis.png")
    save_overlay_image(img_ref_tensor, img_r_result, vis_path)
    print(f"[INFO] 可视化叠图已保存至: {vis_path}")

if __name__ == "__main__":
    main()