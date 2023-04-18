import numpy as np
import torch
import torch.nn as nn
from .models.data_processor import DataProcessor
from .models.mean_vfe import MeanVFE
from .models.spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt
from .models.voxelnext_head import VoxelNeXtHead

from .utils.image_projection import _proj_voxel_image
from segment_anything import SamPredictor, sam_model_registry

class VoxelNeXt(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        point_cloud_range = np.array(model_cfg.POINT_CLOUD_RANGE, dtype=np.float32)

        self.data_processor = DataProcessor(
            model_cfg.DATA_PROCESSOR, point_cloud_range=point_cloud_range,
            training=False, num_point_features=len(model_cfg.USED_FEATURE_LIST)
        )

        input_channels = model_cfg.get('INPUT_CHANNELS', 5)
        grid_size = np.array(model_cfg.get('GRID_SIZE', [1440, 1440, 40]))

        class_names = model_cfg.get('CLASS_NAMES')
        kernel_size_head = model_cfg.get('KERNEL_SIZE_HEAD', 1)
        self.point_cloud_range = torch.Tensor(model_cfg.get('POINT_CLOUD_RANGE', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]))
        self.voxel_size = torch.Tensor(model_cfg.get('VOXEL_SIZE', [0.075, 0.075, 0.2]))
        CLASS_NAMES_EACH_HEAD = model_cfg.get('CLASS_NAMES_EACH_HEAD')
        SEPARATE_HEAD_CFG = model_cfg.get('SEPARATE_HEAD_CFG')
        POST_PROCESSING = model_cfg.get('POST_PROCESSING')
        self.voxelization = MeanVFE()
        self.backbone_3d = VoxelResBackBone8xVoxelNeXt(input_channels, grid_size)
        self.dense_head = VoxelNeXtHead(class_names, self.point_cloud_range, self.voxel_size, kernel_size_head,
                 CLASS_NAMES_EACH_HEAD, SEPARATE_HEAD_CFG, POST_PROCESSING)


class Model(nn.Module):
    def __init__(self, model_cfg, device="cuda"):
        super().__init__()

        sam_type = model_cfg.get('SAM_TYPE', "vit_b")
        sam_checkpoint = model_cfg.get('SAM_CHECKPOINT', "/data/sam_vit_b_01ec64.pth")

        sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint).to(device=device)
        self.sam_predictor = SamPredictor(sam)

        voxelnext_checkpoint = model_cfg.get('VOXELNEXT_CHECKPOINT', "/data/voxelnext_nuscenes_kernel1.pth")
        model_dict = torch.load(voxelnext_checkpoint)
        self.voxelnext = VoxelNeXt(model_cfg).to(device=device)
        self.voxelnext.load_state_dict(model_dict)
        self.point_features = {}
        self.device = device

    def image_embedding(self, image):
        self.sam_predictor.set_image(image)

    def point_embedding(self, data_dict, image_id):
        data_dict = self.voxelnext.data_processor.forward(
            data_dict=data_dict
        )
        data_dict['voxels'] = torch.Tensor(data_dict['voxels']).to(self.device)
        data_dict['voxel_num_points'] = torch.Tensor(data_dict['voxel_num_points']).to(self.device)
        data_dict['voxel_coords'] = torch.Tensor(data_dict['voxel_coords']).to(self.device)

        data_dict = self.voxelnext.voxelization(data_dict)
        n_voxels = data_dict['voxel_coords'].shape[0]
        device = data_dict['voxel_coords'].device
        dtype = data_dict['voxel_coords'].dtype
        data_dict['voxel_coords'] = torch.cat([torch.zeros((n_voxels, 1), device=device, dtype=dtype), data_dict['voxel_coords']], dim=1)
        data_dict['batch_size'] = 1

        if not image_id in self.point_features:
            data_dict = self.voxelnext.backbone_3d(data_dict)
            self.point_features[image_id] = data_dict
        else:
            data_dict = self.point_features[image_id]
        pred_dicts = self.voxelnext.dense_head(data_dict)

        voxel_coords = data_dict['out_voxels'][pred_dicts[0]['voxel_ids'].squeeze(-1)] * self.voxelnext.dense_head.feature_map_stride

        return pred_dicts, voxel_coords

    def generate_3D_box(self, lidar2img_rt, mask, voxel_coords, pred_dicts, quality_score=0.1):
        device = voxel_coords.device
        points_image, depth = _proj_voxel_image(voxel_coords, lidar2img_rt, self.voxelnext.voxel_size.to(device), self.voxelnext.point_cloud_range.to(device))
        points = points_image.permute(1, 0).int().cpu().numpy()
        selected_voxels = torch.zeros_like(depth).squeeze(0)

        for i in range(points.shape[0]):
            point = points[i]
            if point[0] < 0 or point[1] < 0 or point[0] >= mask.shape[1] or point[1] >= mask.shape[0]:
                continue
            if mask[point[1], point[0]]:
                selected_voxels[i] = 1

        mask_extra = (pred_dicts[0]['pred_scores'] > quality_score)
        if mask_extra.sum() == 0:
            print("no high quality 3D box related.")
            return None

        selected_voxels *= mask_extra
        if selected_voxels.sum() > 0:
            selected_box_id = pred_dicts[0]['pred_scores'][selected_voxels.bool()].argmax()
            selected_box = pred_dicts[0]['pred_boxes'][selected_voxels.bool()][selected_box_id]
        else:
            grid_x, grid_y = torch.meshgrid(torch.arange(mask.shape[0]), torch.arange(mask.shape[1]))
            mask_x, mask_y = grid_x[mask], grid_y[mask]
            mask_center = torch.Tensor([mask_y.float().mean(), mask_x.float().mean()]).to(
                pred_dicts[0]['pred_boxes'].device).unsqueeze(1)

            dist = ((points_image - mask_center) ** 2).sum(0)
            selected_id = dist[mask_extra].argmin()
            selected_box = pred_dicts[0]['pred_boxes'][mask_extra][selected_id]
        return selected_box

    def forward(self, image, point_dict, prompt_point, lidar2img_rt, image_id, quality_score=0.1):
        self.image_embedding(image)
        pred_dicts, voxel_coords = self.point_embedding(point_dict, image_id)

        masks, scores, _ = self.sam_predictor.predict(point_coords=prompt_point, point_labels=np.array([1]))
        mask = masks[0]

        box3d = self.generate_3D_box(lidar2img_rt, mask, voxel_coords, pred_dicts, quality_score=quality_score)
        return mask, box3d


if __name__ == '__main__':
    cfg_dataset = 'nuscenes_dataset.yaml'
    cfg_model = 'config.yaml'

    dataset_cfg = cfg_from_yaml_file(cfg_dataset, cfg)
    model_cfg = cfg_from_yaml_file(cfg_model, cfg)

    nuscenes_dataset = NuScenesDataset(dataset_cfg)
    model = Model(model_cfg)

    index = 0
    data_dict = nuscenes_dataset._get_points(index)
    model.point_embedding(data_dict)

