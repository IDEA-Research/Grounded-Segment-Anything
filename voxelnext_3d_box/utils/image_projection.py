import torch
import numpy as np
import cv2

def get_data_info(info, cam_type):

    cam_info = info[cam_type]

    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
    lidar2cam_t = cam_info[
                      'sensor2lidar_translation'] @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    intrinsic = cam_info['cam_intrinsic']
    viewpad = np.eye(4)
    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
    lidar2img_rt = (viewpad @ lidar2cam_rt.T)

    return lidar2img_rt


def _proj_voxel_image(voxel_coords, lidar2img_rt, voxel_size, point_cloud_range):
    # voxel_coords [n ,4]
    # lidar2img_rt [4, 4]
    #x_input.indices [n, 4] [[0, Z, Y, Z]xn]

    voxel_coords = voxel_coords[:, [3,2,1]]
    device = voxel_coords.device
    lidar2img_rt = torch.Tensor(lidar2img_rt).to(device)
    # point_cloud_rangetensor([-51.2000, -51.2000,  -5.0000,  51.2000,  51.2000,   3.0000])
    voxel_coords = voxel_coords * voxel_size.unsqueeze(0) + point_cloud_range[:3].unsqueeze(0)
    # (n, 4)
    voxel_coords = torch.cat([voxel_coords, torch.ones((voxel_coords.shape[0], 1), device=device)], dim=-1)
    points_image = torch.matmul(lidar2img_rt, voxel_coords.permute(1, 0)) #(voxel_coords @ lidar2img_rt).T
    # (4, n)
    depth = points_image[2:3] # (1, n)
    points_image = points_image[:2] / torch.maximum(depth, torch.ones_like(depth*1e-4))
    return points_image, depth

def _draw_image(points_image, image_path, depth):
    image = cv2.imread(image_path)
    points_image = points_image.int().cpu().numpy()
    for i in range(points_image.shape[1]):
        _point = points_image[:, i]
        if _point[0] > 0 and _point[1] > 0 and depth[0][i] >0:
            cv2.circle(image, tuple(_point), 1, (0,255,0), -1)
    #cv2.imwrite("image.png", image)
    return image

def _draw_mask(image_path, mask, color=None):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    if color is None:
        color = np.random.random(3)

    image[mask] = image[mask] * color
    #cv2.imwrite("image_mask.png", image)
    return image

def _draw_3dbox(box, lidar2img_rt, image, mask=None, color=None, output_path="image_box.png"):
    #image = cv2.imread(image_path)
    h, w, _ = image.shape

    if color is None:
        color = np.random.random(3)
    if not mask is None:
        image[mask] = image[mask] * color

    center_x, center_y, center_z, H, W, Z, angle = box[:7]
    sin_angle, cos_angle = torch.sin(angle), torch.cos(angle)
    top1 = [center_x - (H/2 * cos_angle + W/2 * sin_angle), center_y - (H/2 * sin_angle + W/2 * cos_angle), center_z + Z/2]
    top2 = [center_x - (H/2 * cos_angle + W/2 * sin_angle), center_y + (H/2 * sin_angle + W/2 * cos_angle), center_z + Z/2]
    top3 = [center_x + (H/2 * cos_angle + W/2 * sin_angle), center_y + (H/2 * sin_angle + W/2 * cos_angle), center_z + Z/2]
    top4 = [center_x + (H/2 * cos_angle + W/2 * sin_angle), center_y - (H/2 * sin_angle + W/2 * cos_angle), center_z + Z/2]

    down1 = [center_x - (H/2 * cos_angle + W/2 * sin_angle), center_y - (H/2 * sin_angle + W/2 * cos_angle), center_z - Z/2]
    down2 = [center_x - (H/2 * cos_angle + W/2 * sin_angle), center_y + (H/2 * sin_angle + W/2 * cos_angle), center_z - Z/2]
    down3 = [center_x + (H/2 * cos_angle + W/2 * sin_angle), center_y + (H/2 * sin_angle + W/2 * cos_angle), center_z - Z/2]
    down4 = [center_x + (H/2 * cos_angle + W/2 * sin_angle), center_y - (H/2 * sin_angle + W/2 * cos_angle), center_z - Z/2]
    points = torch.Tensor([top1, top2, top3, top4, down1, down2, down3, down4, [center_x, center_y, center_z]]) # (8, 3)
    points = torch.cat([points, torch.ones((points.shape[0], 1), device=points.device)], dim=-1)
    points_image = torch.matmul(torch.Tensor(lidar2img_rt).to(points.device), points.permute(1, 0))
    depth = points_image[2:3] # (1, n)
    points_image = points_image[:2] / torch.maximum(depth, torch.ones_like(depth*1e-4))
    points_image = points_image.permute(1, 0).int().cpu().numpy() #(voxel_coords @ lidar2img_rt).T
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

    cv2.circle(image, tuple(points_image[-1]), 3, (0, 255, 0), -1)

    for line in lines:
        cv2.line(image, tuple(points_image[line[0]]), tuple(points_image[line[1]]), tuple(color * 255), 2)
    #cv2.imwrite(output_path, image)
    return image








