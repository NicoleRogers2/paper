import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.renderer import look_at_rotation
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
)
from romatch import roma_outdoor
import math

# ----------------- 保持原样的全局函数 -----------------
_GLOBAL_MATCHER = None
_GLOBAL_MATCHER_TYPE = None

# 增加三个新的模块：几何重投影一致性验证、空间分布均匀化采样、多约束自适应权重融合
def get_global_matcher(model_type="roma", device='cuda'):
    global _GLOBAL_MATCHER, _GLOBAL_MATCHER_TYPE
    if _GLOBAL_MATCHER is None or _GLOBAL_MATCHER_TYPE != model_type:
        if model_type == "roma":
            print("Loading RoMa outdoor model...")
            _GLOBAL_MATCHER = roma_outdoor(device=device)
            _GLOBAL_MATCHER_TYPE = "roma"
    return _GLOBAL_MATCHER

def ensure_rgb_4d(x, device=None):
    if isinstance(x, np.ndarray): x = torch.from_numpy(x)
    if not x.is_floating_point(): x = x.float()
    if device is not None: x = x.to(device)
    if x.dim() == 4:
        if x.shape[1] == 3 or x.shape[1] == 4: out = x[:, :3, :, :]
        else: out = x[..., :3].permute(0, 3, 1, 2).contiguous()
    elif x.dim() == 3:
        if x.shape[0] == 3 or x.shape[0] == 4: out = x[:3, :, :].unsqueeze(0)
        else: out = x[..., :3].permute(2, 0, 1).unsqueeze(0).contiguous()
    return out

# ----------------- 保持原样的 Loss 类 -----------------
class GeometricProjectionLoss(nn.Module):
    def __init__(self, cameras, image_size, device='cuda'):
        super().__init__()
        self.device = device
        self.cameras = cameras
        self.image_size = image_size
        self.raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
        self.rasterizer = MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings)
    
    def get_matches_from_roma(self, render_img, ref_img, conf_threshold=0.5, max_points=1000):
        matcher = get_global_matcher("roma", self.device)
        img0, img1 = ensure_rgb_4d(render_img, self.device), ensure_rgb_4d(ref_img, self.device)
        hs, ws = matcher.h_resized, matcher.w_resized
        orig_h, orig_w = img0.shape[2], img0.shape[3]
        img0_resized = F.interpolate(img0, size=(hs, ws), mode='bilinear', align_corners=False)
        img1_resized = F.interpolate(img1, size=(hs, ws), mode='bilinear', align_corners=False)
        if img0_resized.max() > 1.0: img0_resized = img0_resized / 255.0
        if img1_resized.max() > 1.0: img1_resized = img1_resized / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        img0_norm = (img0_resized - mean) / std
        img1_norm = (img1_resized - mean) / std
        batch = {"im_A": img0_norm, "im_B": img1_norm}
        scale_factor = math.sqrt(hs * ws / (560**2))
        with torch.no_grad():
            matcher.train(False)
            if matcher.symmetric: corresps = matcher.forward_symmetric(batch, batched=True, scale_factor=scale_factor)
            else: corresps = matcher.forward(batch, batched=True, scale_factor=scale_factor)
        flow = corresps[1]["flow"]
        certainty = corresps[1]["certainty"]
        b, _, h, w = flow.shape
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1 + 1/h, 1 - 1/h, h, device=self.device), torch.linspace(-1 + 1/w, 1 - 1/w, w, device=self.device), indexing='ij')
        pts_render_x, pts_render_y = (grid_x + 1) / 2 * orig_w, (grid_y + 1) / 2 * orig_h
        pts_ref_x, pts_ref_y = (flow[0, 0] + 1) / 2 * orig_w, (flow[0, 1] + 1) / 2 * orig_h
        conf = certainty[0, 0].sigmoid()
        mask = conf > conf_threshold
        pts_render = torch.stack([pts_render_x[mask], pts_render_y[mask]], dim=-1)
        pts_ref = torch.stack([pts_ref_x[mask], pts_ref_y[mask]], dim=-1)
        confidence = conf[mask]
        if pts_render.shape[0] > max_points:
            indices = torch.randperm(pts_render.shape[0])[:max_points]
            pts_render, pts_ref, confidence = pts_render[indices], pts_ref[indices], confidence[indices]
        return pts_render.detach(), pts_ref.detach(), confidence.detach()
    
    def pixel_to_ndc(self, pts_pixel, image_size):
        H, W = image_size
        x_ndc = -((pts_pixel[:, 0] / W) * 2 - 1)
        y_ndc = -((pts_pixel[:, 1] / H) * 2 - 1)
        return torch.stack([x_ndc, y_ndc], dim=-1)
    
    def unproject_points(self, pts_pixel, depth_map, cameras):
        H, W = depth_map.shape[1], depth_map.shape[2]
        x_idx = pts_pixel[:, 0].long().clamp(0, W-1)
        y_idx = pts_pixel[:, 1].long().clamp(0, H-1)
        depths = depth_map[0, y_idx, x_idx]
        valid_mask = depths > 0
        if valid_mask.sum() == 0: return None, None
        pts_pixel_valid, depths_valid = pts_pixel[valid_mask], depths[valid_mask]
        pts_ndc = self.pixel_to_ndc(pts_pixel_valid, (H, W))
        pts_ndc_3d = torch.cat([pts_ndc, depths_valid.unsqueeze(-1)], dim=-1)
        pts_3d = cameras.unproject_points(pts_ndc_3d, world_coordinates=True)
        return pts_3d, valid_mask
    
    def project_points(self, pts_3d, cameras, image_size):
        pts_ndc = cameras.transform_points_ndc(pts_3d)
        H, W = image_size
        x_pixel = (-pts_ndc[:, 0] + 1) / 2 * W
        y_pixel = (-pts_ndc[:, 1] + 1) / 2 * H
        return torch.stack([x_pixel, y_pixel], dim=-1)

# ----------------- 优化过的前向传播 -----------------
class GeometricTModel(nn.Module):
    def __init__(self, meshes, rasterizer, phong_renderer, cameras, image_ref, t_start, device='cuda',
                 tau_reproj=5.0, grid_h=6, grid_w=10, max_per_cell=3, 
                 lambda1=2.0, lambda2=1.0, lambda3=1.0):
        super().__init__()
        self.meshes, self.device = meshes, device
        self.rasterizer, self.phong_renderer, self.cameras = rasterizer, phong_renderer, cameras
        if isinstance(image_ref, np.ndarray): self.register_buffer('image_ref', torch.from_numpy(image_ref))
        else: self.register_buffer('image_ref', image_ref.clone().detach())
        
        self.camera_position = torch.tensor([0.0, 0.0, -0.2], dtype=torch.float32, device=self.device)
        self.R = look_at_rotation(self.camera_position[None, :], device=self.device)
        self.T_cam = -torch.bmm(self.R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]
        
        t_start_tensor = torch.tensor(t_start, dtype=torch.float32, device=self.device).view(-1) if not isinstance(t_start, torch.Tensor) else t_start.clone().detach().float().to(self.device).view(-1)
        self.t_x = nn.Parameter(t_start_tensor[0].clone())
        self.t_y = nn.Parameter(t_start_tensor[1].clone())
        self.t_z = nn.Parameter(t_start_tensor[2].clone())
        
        image_size = (image_ref.shape[0], image_ref.shape[1]) if isinstance(image_ref, np.ndarray) else (image_ref.shape[1] if image_ref.dim() == 4 else image_ref.shape[0], image_ref.shape[2] if image_ref.dim() == 4 else image_ref.shape[1])
        self.geo_loss = GeometricProjectionLoss(cameras, image_size, device)
        
        self.cached_pts_3d_local = None
        self.cached_pts_ref = None
        self.cache_step = 0
        self.last_rendered_image = None 
        
        # 新增超参数和状态变量
        self.tau_reproj = tau_reproj
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.max_per_cell = max_per_cell
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.prev_geo_loss = None
        self.cached_confidence_filtered = None

    def get_current_translation(self): return torch.stack([self.t_x, self.t_y, self.t_z])
    def get_current_mesh(self): return Meshes(verts=self.meshes.verts_padded() + self.get_current_translation(), faces=self.meshes.faces_padded(), textures=self.meshes.textures)
    def render_and_get_depth(self, mesh):
        image = self.phong_renderer(meshes_world=mesh, R=self.R, T=self.T_cam)
        depth = self.rasterizer(mesh, R=self.R, T=self.T_cam).zbuf[..., 0]
        return image, depth
    
    def forward(self, update_cache=True, cache_interval=5):
        should_update = update_cache and (self.cached_pts_3d_local is None or self.cache_step % cache_interval == 0)
        
        if should_update:
            current_mesh = self.get_current_mesh()
            rendered_image, depth_map = self.render_and_get_depth(current_mesh)
            rendered_image = self._image_crop(rendered_image)
            depth_map = self._depth_crop(depth_map)
            
            self.last_rendered_image = rendered_image.detach()
            
            with torch.no_grad():
                pts_render, pts_ref, confidence = self.geo_loss.get_matches_from_roma(rendered_image.detach(), self.image_ref, conf_threshold=0.3, max_points=500)
                
            if pts_render.shape[0] < 10:
                self.cache_step += 1
                return self._fallback_loss(rendered_image), self.last_rendered_image, {'num_matches': pts_render.shape[0], 'using_fallback': True}
                
            with torch.no_grad():
                pts_3d, valid_mask = self.geo_loss.unproject_points(pts_render, depth_map.detach(), self.cameras)
                
            if pts_3d is None or pts_3d.shape[0] < 10:
                self.cache_step += 1
                return self._fallback_loss(rendered_image), self.last_rendered_image, {'num_matches': 0, 'using_fallback': True}
            
            # 【模块1：几何重投影一致性验证】
            with torch.no_grad():
                H_d, W_d = depth_map.shape[1], depth_map.shape[2]
                u_reproj = self.geo_loss.project_points(pts_3d, self.cameras, (H_d, W_d))
                pts_render_valid = pts_render[valid_mask]
                e_reproj = torch.norm(u_reproj - pts_render_valid, dim=-1)
                reproj_mask = e_reproj < self.tau_reproj
                
                pts_3d = pts_3d[reproj_mask]
                pts_ref_valid = pts_ref[valid_mask][reproj_mask]
                conf_valid = confidence[valid_mask][reproj_mask]
                
                if pts_3d.shape[0] < 10:
                    self.cache_step += 1
                    return self._fallback_loss(rendered_image), self.last_rendered_image, {'num_matches': 0, 'using_fallback': True}

            # 【模块2：空间分布均匀化采样】
            with torch.no_grad():
                pts_x = pts_ref_valid[:, 0].clamp(0, W_d - 1)
                pts_y = pts_ref_valid[:, 1].clamp(0, H_d - 1)
                
                cell_w = W_d / self.grid_w
                cell_h = H_d / self.grid_h
                
                bin_x = (pts_x / cell_w).long().clamp(0, self.grid_w - 1)
                bin_y = (pts_y / cell_h).long().clamp(0, self.grid_h - 1)
                bin_idx = bin_y * self.grid_w + bin_x
                
                selected_indices = []
                for b in torch.unique(bin_idx):
                    mask_b = (bin_idx == b)
                    idx_b = torch.where(mask_b)[0]
                    conf_b = conf_valid[idx_b]
                    # 按置信度排序
                    sorted_idx = torch.argsort(conf_b, descending=True)
                    keep_idx = idx_b[sorted_idx[:self.max_per_cell]]
                    selected_indices.append(keep_idx)
                
                if len(selected_indices) > 0:
                    selected_indices = torch.cat(selected_indices)
                    pts_3d = pts_3d[selected_indices]
                    pts_ref_valid = pts_ref_valid[selected_indices]
                    self.cached_confidence_filtered = conf_valid[selected_indices].detach().clone()
                else:
                    self.cache_step += 1
                    return self._fallback_loss(rendered_image), self.last_rendered_image, {'num_matches': 0, 'using_fallback': True}

            current_translation = self.get_current_translation().detach()
            self.cached_pts_3d_local = (pts_3d - current_translation).detach().clone()
            self.cached_pts_ref = pts_ref_valid.detach().clone()
        
        self.cache_step += 1
        if self.cached_pts_3d_local is None:
            return self._fallback_loss(self.last_rendered_image), self.last_rendered_image, {'num_matches': 0, 'using_fallback': True}
        
        translation = self.get_current_translation()
        pts_3d_transformed = self.cached_pts_3d_local + translation
        pts_projected = self.geo_loss.project_points(pts_3d_transformed, self.cameras, (546, 966))
        
        reprojection_error = pts_projected - self.cached_pts_ref
        loss = (reprojection_error ** 2).sum(dim=-1).mean() / (546 * 966) * 100
        
        # 【模块3：多约束自适应权重融合】
        with torch.no_grad():
            img_render_4d = ensure_rgb_4d(self.last_rendered_image, self.device)
            img_ref_4d = ensure_rgb_4d(self.image_ref, self.device)
            L_photo = F.mse_loss(img_render_4d, img_ref_4d).item()
            
            c_mean = self.cached_confidence_filtered.mean().item() if self.cached_confidence_filtered is not None else 0.5
            
            current_geo_loss = loss.item()
            if self.prev_geo_loss is not None and self.prev_geo_loss > 1e-8:
                delta_r = (self.prev_geo_loss - current_geo_loss) / (self.prev_geo_loss + 1e-8)
                # 限制 delta_r 的下限，防止损失突增导致极其负的 delta_r 引发数值溢出
                # 即使 loss 变差 10 倍，delta_r 最低被截断在 -10.0，合理且稳定
                delta_r = max(delta_r, -10.0)
            else:
                delta_r = 0.0
            self.prev_geo_loss = current_geo_loss
            
            if should_update:
                v = (depth_map > 0).float().sum().item() / (depth_map.shape[1] * depth_map.shape[2])
                self._last_v = v
            else:
                v = getattr(self, '_last_v', 0.5)
            
            # 使用截断防溢出计算 exp
            exponent = -(self.lambda1 * c_mean + self.lambda2 * delta_r + self.lambda3 * v)
            # 将指数限制在 [-80, 80] 的安全范围内，math.exp(80) 约等于 5.5e34，完全不会溢出
            exponent = max(min(exponent, 80.0), -80.0)
            alpha = 1.0 / (1.0 + math.exp(exponent))
            
        loss = alpha * loss
        
        loss_info = {
            'reprojection_error': current_geo_loss, 
            'num_matches': self.cached_pts_3d_local.shape[0], 
            'mean_pixel_error': reprojection_error.abs().mean().item(), 
            'using_fallback': False,
            'alpha': alpha,
            'L_photo': L_photo,
            'c_mean': c_mean,
            'delta_r': delta_r,
            'visibility': v
        }
        return loss, self.last_rendered_image, loss_info
    
    def _fallback_loss(self, rendered_image): return F.mse_loss(ensure_rgb_4d(rendered_image, self.device), ensure_rgb_4d(self.image_ref, self.device)) * 10
    def _image_crop(self, image):
        H, W = image.shape[1], image.shape[2]; target_h, target_w = 546, 966
        return image[:, (H - target_h) // 2:(H - target_h) // 2 + target_h, (W - target_w) // 2:(W - target_w) // 2 + target_w, :] if H >= target_h and W >= target_w else image
    def _depth_crop(self, depth):
        H, W = depth.shape[1], depth.shape[2]; target_h, target_w = 546, 966
        return depth[:, (H - target_h) // 2:(H - target_h) // 2 + target_h, (W - target_w) // 2:(W - target_w) // 2 + target_w] if H >= target_h and W >= target_w else depth
    def get_translation(self): return [self.t_x.item(), self.t_y.item(), self.t_z.item()]


class GeometricRModel(nn.Module):
    def __init__(self, meshes, rasterizer, phong_renderer, cameras, image_ref, t_result, device='cuda',
                 tau_reproj=5.0, grid_h=6, grid_w=10, max_per_cell=3,
                 lambda1=2.0, lambda2=1.0, lambda3=1.0):
        super().__init__()
        self.meshes, self.device = meshes, device
        self.rasterizer, self.phong_renderer, self.cameras = rasterizer, phong_renderer, cameras
        if isinstance(image_ref, np.ndarray): self.register_buffer('image_ref', torch.from_numpy(image_ref))
        else: self.register_buffer('image_ref', image_ref.clone().detach())
        if isinstance(t_result, np.ndarray): self.register_buffer('t_result', torch.from_numpy(t_result).float().view(-1))
        elif isinstance(t_result, torch.Tensor): self.register_buffer('t_result', t_result.clone().detach().float().view(-1))
        else: self.register_buffer('t_result', torch.tensor(t_result, dtype=torch.float32).view(-1))
        
        self.camera_position = torch.tensor([0.0, 0.0, -0.2], dtype=torch.float32, device=self.device)
        self.R = look_at_rotation(self.camera_position[None, :], device=self.device)
        self.T_cam = -torch.bmm(self.R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]
        
        self.rotation_quat = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device))
        image_size = (image_ref.shape[0], image_ref.shape[1]) if isinstance(image_ref, np.ndarray) else (image_ref.shape[1] if image_ref.dim() == 4 else image_ref.shape[0], image_ref.shape[2] if image_ref.dim() == 4 else image_ref.shape[1])
        self.geo_loss = GeometricProjectionLoss(cameras, image_size, device)
        
        self.cached_pts_3d_local = None
        self.cached_pts_ref = None
        self.cache_step = 0
        self.last_rendered_image = None
        
        # 新增超参数和状态变量
        self.tau_reproj = tau_reproj
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.max_per_cell = max_per_cell
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.prev_geo_loss = None
        self.cached_confidence_filtered = None
    
    def get_rotation_matrix(self): return quaternion_to_matrix(F.normalize(self.rotation_quat, p=2, dim=0))
    def get_current_mesh(self): return Meshes(verts=torch.matmul(self.meshes.verts_padded(), self.get_rotation_matrix()) + self.t_result, faces=self.meshes.faces_padded(), textures=self.meshes.textures)
    def render_and_get_depth(self, mesh):
        image = self.phong_renderer(meshes_world=mesh, R=self.R, T=self.T_cam)
        depth = self.rasterizer(mesh, R=self.R, T=self.T_cam).zbuf[..., 0]
        return image, depth
    
    def forward(self, update_cache=True, cache_interval=5):
        should_update = update_cache and (self.cached_pts_3d_local is None or self.cache_step % cache_interval == 0)
        
        if should_update:
            current_mesh = self.get_current_mesh()
            rendered_image, depth_map = self.render_and_get_depth(current_mesh)
            rendered_image = self._image_crop(rendered_image)
            depth_map = self._depth_crop(depth_map)
            
            self.last_rendered_image = rendered_image.detach()
            
            with torch.no_grad():
                pts_render, pts_ref, confidence = self.geo_loss.get_matches_from_roma(rendered_image.detach(), self.image_ref, conf_threshold=0.3, max_points=500)
                
            if pts_render.shape[0] < 10:
                self.cache_step += 1
                return self._fallback_loss(rendered_image), self.last_rendered_image, {'num_matches': pts_render.shape[0], 'using_fallback': True}
                
            with torch.no_grad():
                pts_3d, valid_mask = self.geo_loss.unproject_points(pts_render, depth_map.detach(), self.cameras)
                
            if pts_3d is None or pts_3d.shape[0] < 10:
                self.cache_step += 1
                return self._fallback_loss(rendered_image), self.last_rendered_image, {'num_matches': 0, 'using_fallback': True}
            
            # 【模块1：几何重投影一致性验证】
            with torch.no_grad():
                H_d, W_d = depth_map.shape[1], depth_map.shape[2]
                u_reproj = self.geo_loss.project_points(pts_3d, self.cameras, (H_d, W_d))
                pts_render_valid = pts_render[valid_mask]
                e_reproj = torch.norm(u_reproj - pts_render_valid, dim=-1)
                reproj_mask = e_reproj < self.tau_reproj
                
                pts_3d = pts_3d[reproj_mask]
                pts_ref_valid = pts_ref[valid_mask][reproj_mask]
                conf_valid = confidence[valid_mask][reproj_mask]
                
                if pts_3d.shape[0] < 10:
                    self.cache_step += 1
                    return self._fallback_loss(rendered_image), self.last_rendered_image, {'num_matches': 0, 'using_fallback': True}

            # 【模块2：空间分布均匀化采样】
            with torch.no_grad():
                pts_x = pts_ref_valid[:, 0].clamp(0, W_d - 1)
                pts_y = pts_ref_valid[:, 1].clamp(0, H_d - 1)
                
                cell_w = W_d / self.grid_w
                cell_h = H_d / self.grid_h
                
                bin_x = (pts_x / cell_w).long().clamp(0, self.grid_w - 1)
                bin_y = (pts_y / cell_h).long().clamp(0, self.grid_h - 1)
                bin_idx = bin_y * self.grid_w + bin_x
                
                selected_indices = []
                for b in torch.unique(bin_idx):
                    mask_b = (bin_idx == b)
                    idx_b = torch.where(mask_b)[0]
                    conf_b = conf_valid[idx_b]
                    sorted_idx = torch.argsort(conf_b, descending=True)
                    keep_idx = idx_b[sorted_idx[:self.max_per_cell]]
                    selected_indices.append(keep_idx)
                
                if len(selected_indices) > 0:
                    selected_indices = torch.cat(selected_indices)
                    pts_3d = pts_3d[selected_indices]
                    pts_ref_valid = pts_ref_valid[selected_indices]
                    self.cached_confidence_filtered = conf_valid[selected_indices].detach().clone()
                else:
                    self.cache_step += 1
                    return self._fallback_loss(rendered_image), self.last_rendered_image, {'num_matches': 0, 'using_fallback': True}

            rotation_matrix = self.get_rotation_matrix().detach()
            self.cached_pts_3d_local = torch.matmul(pts_3d - self.t_result, rotation_matrix.T).detach().clone()
            self.cached_pts_ref = pts_ref_valid.detach().clone()
        
        self.cache_step += 1
        if self.cached_pts_3d_local is None:
            return self._fallback_loss(self.last_rendered_image), self.last_rendered_image, {'num_matches': 0, 'using_fallback': True}
        
        pts_3d_transformed = torch.matmul(self.cached_pts_3d_local, self.get_rotation_matrix()) + self.t_result
        pts_projected = self.geo_loss.project_points(pts_3d_transformed, self.cameras, (546, 966))
        
        reprojection_error = pts_projected - self.cached_pts_ref
        loss = (reprojection_error ** 2).sum(dim=-1).mean() / (546 * 966) * 100
        
        quat_normalized = F.normalize(self.rotation_quat, p=2, dim=0)
        angle_deg = 2 * torch.acos(torch.clamp(quat_normalized[0].abs(), -1, 1)) * 180 / torch.pi
        
        # 【模块3：多约束自适应权重融合】
        with torch.no_grad():
            img_render_4d = ensure_rgb_4d(self.last_rendered_image, self.device)
            img_ref_4d = ensure_rgb_4d(self.image_ref, self.device)
            L_photo = F.mse_loss(img_render_4d, img_ref_4d).item()
            
            c_mean = self.cached_confidence_filtered.mean().item() if self.cached_confidence_filtered is not None else 0.5
            
            current_geo_loss = loss.item()
            if self.prev_geo_loss is not None and self.prev_geo_loss > 1e-8:
                delta_r = (self.prev_geo_loss - current_geo_loss) / (self.prev_geo_loss + 1e-8)
                # 限制 delta_r 的下限，防止损失突增导致极其负的 delta_r 引发数值溢出
                # 即使 loss 变差 10 倍，delta_r 最低被截断在 -10.0，合理且稳定
                delta_r = max(delta_r, -10.0)
            else:
                delta_r = 0.0
            self.prev_geo_loss = current_geo_loss
            
            if should_update:
                v = (depth_map > 0).float().sum().item() / (depth_map.shape[1] * depth_map.shape[2])
                self._last_v = v
            else:
                v = getattr(self, '_last_v', 0.5)
            
            # 使用截断防溢出计算 exp
            exponent = -(self.lambda1 * c_mean + self.lambda2 * delta_r + self.lambda3 * v)
            # 将指数限制在 [-80, 80] 的安全范围内，math.exp(80) 约等于 5.5e34，完全不会溢出
            exponent = max(min(exponent, 80.0), -80.0)
            alpha = 1.0 / (1.0 + math.exp(exponent))
            
        loss = alpha * loss
        
        loss_info = {
            'reprojection_error': current_geo_loss, 
            'num_matches': self.cached_pts_3d_local.shape[0], 
            'mean_pixel_error': reprojection_error.abs().mean().item(), 
            'rotation_angle_deg': angle_deg.item(),
            'using_fallback': False,
            'alpha': alpha,
            'L_photo': L_photo,
            'c_mean': c_mean,
            'delta_r': delta_r,
            'visibility': v
        }
        return loss, self.last_rendered_image, loss_info

    def _fallback_loss(self, rendered_image): return F.mse_loss(ensure_rgb_4d(rendered_image, self.device), ensure_rgb_4d(self.image_ref, self.device)) * 10
    def _image_crop(self, image):
        H, W = image.shape[1], image.shape[2]; target_h, target_w = 546, 966
        return image[:, (H - target_h) // 2:(H - target_h) // 2 + target_h, (W - target_w) // 2:(W - target_w) // 2 + target_w, :] if H >= target_h and W >= target_w else image
    def _depth_crop(self, depth):
        H, W = depth.shape[1], depth.shape[2]; target_h, target_w = 546, 966
        return depth[:, (H - target_h) // 2:(H - target_h) // 2 + target_h, (W - target_w) // 2:(W - target_w) // 2 + target_w] if H >= target_h and W >= target_w else depth
    def get_rotation_quat(self): return F.normalize(self.rotation_quat, p=2, dim=0)