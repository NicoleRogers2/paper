import os
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.transforms import (
    quaternion_to_matrix,
    matrix_to_euler_angles,
    euler_angles_to_matrix,
)
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    DirectionalLights,
    TexturesVertex,
)

# v2 model
from p3d_model_with_reprojection_v4 import GeometricTModel, GeometricRModel


# ============================================================
# Adaptive LR Scheduler
# ============================================================

class AdaptiveLRScheduler:

    def __init__(
        self,
        base_lr_t=0.01,
        base_lr_r=0.01,
        patience=5,
        min_delta=1e-4,
        max_iterations=30,
        min_lr=1e-6,
        max_lr=0.1,
    ):
        self.base_lr_t = base_lr_t
        self.base_lr_r = base_lr_r
        self.patience = patience
        self.min_delta = min_delta
        self.max_iterations = max_iterations
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.reset()

    def reset(self):
        self.loss_history = []
        self.no_improve_count = 0
        self.best_loss = float("inf")

    def get_translation_lr(self, pixel_error, iteration):
        scale = max(min(pixel_error / 10.0, 2.0), 0.1)
        decay = 0.95 ** (iteration // 10)
        lr = self.base_lr_t * scale * decay
        return max(self.min_lr, min(lr, self.max_lr))

    def get_rotation_lr(self, pixel_error, current_angle, iteration):

        if current_angle > 60:
            angle_factor = 0.2
        elif current_angle > 30:
            angle_factor = 0.5
        else:
            angle_factor = 1.0

        scale = max(min(pixel_error / 10.0, 2.0), 0.1)
        decay = 0.95 ** (iteration // 10)
        lr = self.base_lr_r * scale * angle_factor * decay

        return max(self.min_lr, min(lr, self.max_lr))

    def should_stop(self, current_loss):

        self.loss_history.append(current_loss)

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        return (
            self.no_improve_count >= self.patience
            or len(self.loss_history) >= self.max_iterations
            or current_loss < 0.001
        )

    def get_convergence_info(self):
        return {
            "iterations": len(self.loss_history),
            "best_loss": self.best_loss,
            "final_loss": self.loss_history[-1] if self.loss_history else None,
            "converged": self.no_improve_count >= self.patience,
        }


# ============================================================
# Utility functions
# ============================================================

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_camera(device):
    return FoVPerspectiveCameras(
        fov=40.2,
        znear=0.1,
        zfar=1000.0,
        device=device,
    )


def create_raster_settings():
    return RasterizationSettings(
        image_size=(549, 976),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        max_faces_per_bin=200000,
    )


def create_rasterizer(device, cameras, raster_settings):
    return MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings,
    )


def create_phong_renderer(device, cameras, raster_settings):

    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings,
    )

    lights = DirectionalLights(
        device=device,
        direction=((0, 0, -5),),
    )

    return MeshRenderer(
        rasterizer=rasterizer,
        shader=HardPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
        ),
    )


def load_image_as_tensor(path, device, size=(549, 976)):

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)

    alpha = np.ones((size[1], size[0], 1), dtype=np.uint8) * 255
    img_rgba = np.concatenate([img, alpha], axis=-1)

    tensor = torch.from_numpy(img_rgba).float().unsqueeze(0).to(device) / 255.0
    return tensor


def load_head_mesh(data_dir, obj_file, device):

    verts, faces, _ = load_obj(
        os.path.join(data_dir, obj_file),
        load_textures=False,
    )

    verts_rgb = torch.ones_like(verts)

    mesh = Meshes(
        verts=[verts],
        faces=[faces.verts_idx],
        textures=TexturesVertex(verts_features=[verts_rgb]),
    )

    return mesh.to(device)


def image_cropped(image):

    H, W = image.shape[1], image.shape[2]

    if H >= 546 and W >= 966:
        return image[
            :,
            (H - 546) // 2 : (H - 546) // 2 + 546,
            (W - 966) // 2 : (W - 966) // 2 + 966,
            :,
        ]

    return image


def rotate_object_by_angles(rotation_angles):
    return euler_angles_to_matrix(rotation_angles, convention="ZXY")


def final_angles(rotation_params, quat):

    quat = F.normalize(quat, p=2, dim=0)

    base_matrix = quaternion_to_matrix(quat)

    delta_matrix = euler_angles_to_matrix(
        torch.tensor(rotation_params[:3], dtype=torch.float32, device=quat.device)
        * (math.pi / 180.0),
        convention="ZXY",
    )

    final_matrix = base_matrix @ delta_matrix

    angles = matrix_to_euler_angles(final_matrix, convention="ZXY")

    return angles * (180.0 / math.pi)


# ============================================================
# Translation optimization
# ============================================================

def predict_head_pose_t_geometric(
    t_start,
    r_start,
    head_mesh,
    rasterizer,
    phong_renderer,
    cameras,
    img_ref_tensor,
    scheduler,
    device=None,
):

    r_matrix = rotate_object_by_angles(
        torch.tensor(r_start[:3], dtype=torch.float32, device=device)
        * (torch.pi / 180)
    )

    t_head_mesh = Meshes(
        verts=torch.matmul(head_mesh.verts_padded(), r_matrix),
        faces=head_mesh.faces_padded(),
        textures=head_mesh.textures,
    )

    img_ref_np = img_ref_tensor.cpu().detach().numpy().squeeze()[..., :3]

    model = GeometricTModel(
        t_head_mesh,
        rasterizer,
        phong_renderer,
        cameras,
        img_ref_np,
        np.array(t_start, dtype=np.float32),
        device,
    ).to(device)

    scheduler.reset()
    optimizer = torch.optim.Adam(model.parameters(), lr=scheduler.base_lr_t)

    image_init = None
    iteration = 0
    early_stop_flag = False

    while True:

        optimizer.zero_grad()

        loss, rendered_image, loss_info = model.forward(
            update_cache=True,
            cache_interval=10,
        )

        if image_init is None and rendered_image is not None:
            image_init = rendered_image.clone().detach()

        if scheduler.should_stop(loss.item()):
            if iteration <= 5:
                early_stop_flag = True
            break

        loss.backward()

        for param_group in optimizer.param_groups:
            param_group["lr"] = scheduler.get_translation_lr(
                loss_info.get("mean_pixel_error", 10),
                iteration,
            )

        optimizer.step()
        iteration += 1

    with torch.no_grad():
        _, final_img, _ = model.forward(update_cache=False)

    return (
        model.get_translation(),
        image_init,
        final_img,
        scheduler.get_convergence_info(),
        early_stop_flag,
    )


# ============================================================
# Rotation optimization
# ============================================================

def predict_head_pose_r_geometric(
    t_result,
    r_start,
    head_mesh,
    rasterizer,
    phong_renderer,
    cameras,
    img_ref_tensor,
    scheduler,
    device=None,
):

    r_matrix = rotate_object_by_angles(
        torch.tensor(r_start[:3], dtype=torch.float32, device=device)
        * (torch.pi / 180)
    )

    r_head_mesh = Meshes(
        verts=torch.matmul(head_mesh.verts_padded(), r_matrix),
        faces=head_mesh.faces_padded(),
        textures=head_mesh.textures,
    )

    img_ref_np = img_ref_tensor.cpu().detach().numpy().squeeze()[..., :3]

    model = GeometricRModel(
        r_head_mesh,
        rasterizer,
        phong_renderer,
        cameras,
        img_ref_np,
        t_result,
        device,
    ).to(device)

    scheduler.reset()
    optimizer = torch.optim.Adam(model.parameters(), lr=scheduler.base_lr_r)

    image_init = None
    iteration = 0
    early_stop_flag = False

    while True:

        optimizer.zero_grad()

        loss, rendered_image, loss_info = model.forward(
            update_cache=True,
            cache_interval=10,
        )

        if image_init is None and rendered_image is not None:
            image_init = rendered_image.clone().detach()

        if scheduler.should_stop(loss.item()):
            if iteration <= 5:
                early_stop_flag = True
            break

        loss.backward()

        for param_group in optimizer.param_groups:
            param_group["lr"] = scheduler.get_rotation_lr(
                loss_info.get("mean_pixel_error", 10),
                loss_info.get("rotation_angle_deg", 0),
                iteration,
            )

        optimizer.step()
        iteration += 1

    with torch.no_grad():
        _, final_img, _ = model.forward(update_cache=False)

    return (
        model.get_rotation_quat(),
        image_init,
        final_img,
        scheduler.loss_history[-1]
        if scheduler.loss_history
        else float("inf"),
        scheduler.get_convergence_info(),
        early_stop_flag,
    )