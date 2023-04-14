import warnings

import mmcv
import numpy as np
from mmcv.image import imwrite
from mmcv.visualization.image import imshow

from mmpose.core import imshow_keypoints
from .. import builder
from ..builder import POSENETS
from .base import BasePose
import torch
from config import cfg

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16

from .top_down import TopDown


@POSENETS.register_module()
class Poseur(TopDown):
    def __init__(self, *args, **kwargs):
        if 'filp_fuse_type' in kwargs:
            self.filp_fuse_type = kwargs.pop('filp_fuse_type')
        else:
            self.filp_fuse_type = 'default'
        super().__init__(*args, **kwargs)

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        self.backbone.init_weights(pretrained)
        if self.with_neck:
            self.neck.init_weights()
        if self.with_keypoint:
            self.keypoint_head.init_weights()

    @auto_fp16(apply_to=('img',))
    def forward(self,
                img,
                coord_target=None,
                coord_target_weight=None,
                bbox_target=None,
                bbox_target_weight=None,
                hp_target=None,
                hp_target_weight=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                coord_init=None,
                query_init=None,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C (Default: 3)
            img height: imgH
            img weight: imgW
            heatmaps height: H
            heatmaps weight: W

        Args:
            img (torch.Tensor[NxCximgHximgW]): Input images.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]): Weights across
                different joint types.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.
            return_heatmap (bool) : Option to return heatmap.

        Returns:
            dict|tuple: if `return loss` is true, then return losses.
              Otherwise, return predicted poses, boxes, image paths
                  and heatmaps.
        """
        return self.forward_mesh_recovery(img, coord_init=coord_init, query_init=query_init,
                                          **kwargs)
        # if return_loss:
        #     return self.forward_train(img,
        #                               coord_target, coord_target_weight,
        #                               hp_target, hp_target_weight, img_metas,
        #                               **kwargs)
        # return self.forward_test(
        #     img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward_train(self, img, coord_target, coord_target_weight,
                      hp_target, hp_target_weight, img_metas, **kwargs):
        """
        :param img:
        :param coord_target: [2, 17, 2]
        :param coord_target_weight: [2, 17, 2]
        :param hp_target: [2, 4, 17, 64, 48]
        :param hp_target_weight: [2, 4, 17, 1]
        :param img_metas:
        :param kwargs:
        :return:
        """
        """Defines the computation performed at every call when training."""
        output = self.backbone(img)
        img_feat = output[-1]
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            # output = self.keypoint_head(output, img_metas)
            enc_output, dec_output = self.keypoint_head(output)

        return img_feat, enc_output, dec_output, None

    def seperate_sigma_from_score(self, score):
        if score.shape[2] == 3:
            sigma = score[:, :, [1, 2]]
            score = score[:, :, [0]]
            return score, sigma
        elif score.shape[2] == 1:
            return score, None
        else:
            raise

    def forward_mesh_recovery(self, output, coord_init=None, query_init=None, **kwargs):
        """
        :param img:
        :param coord_target: [2, 17, 2]
        :param coord_target_weight: [2, 17, 2]
        :param hp_target: [2, 4, 17, 64, 48]
        :param hp_target_weight: [2, 4, 17, 1]
        :param img_metas:
        :param kwargs:
        :return:
        """
        """Defines the computation performed at every call when training."""
        # output = self.backbone(img)
        img_feat = output[-1]
        # print(len(output))
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            # output = self.keypoint_head(output, img_metas)
            enc_output, dec_output = self.keypoint_head(output, coord_init=coord_init, query_init=query_init)

            return dec_output.feat[-1]

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]

        result = {}

        features = self.backbone(img)
        if self.with_neck:
            features = self.neck(features)
        if self.with_keypoint:
            output_regression, output_regression_score = self.keypoint_head.inference_model(
                features, flip_pairs=None)
            output_regression_score, output_regression_sigma = self.seperate_sigma_from_score(output_regression_score)

        if self.test_cfg['flip_test']:
            img_flipped = img.flip(3)
            features_flipped = self.backbone(img_flipped)
            if self.with_neck:
                features_flipped = self.neck(features_flipped)
            if self.with_keypoint:
                output_regression_flipped, output_regression_score_flipped = self.keypoint_head.inference_model(
                    features_flipped, img_metas[0]['flip_pairs'])
                output_regression_score_flipped, output_regression_sigma_flipped = \
                    self.seperate_sigma_from_score(output_regression_score_flipped)
                if self.filp_fuse_type == 'default':
                    output_regression = (output_regression +
                                         output_regression_flipped) * 0.5

                    output_regression_score = (output_regression_score +
                                               output_regression_score_flipped) * 0.5
                elif self.filp_fuse_type == 'type1':
                    # output_regression = (output_regression * output_regression_score + output_regression_flipped * output_regression_score_flipped)\
                    #     /(output_regression_score + output_regression_score_flipped+1e-9)
                    output_regression, output_regression_flipped = \
                        torch.from_numpy(output_regression), torch.from_numpy(output_regression_flipped)

                    output_regression_score, output_regression_score_flipped = \
                        torch.from_numpy(output_regression_score), torch.from_numpy(output_regression_score_flipped)

                    output_regression = (
                                                    output_regression * output_regression_score + output_regression_flipped * output_regression_score_flipped) \
                                        / (output_regression_score + output_regression_score_flipped + 1e-9)

                    diff = 1 - (output_regression_score - output_regression_score_flipped).abs()
                    output_regression_score = (output_regression_score * output_regression_score_flipped * diff) ** 2

                    output_regression = output_regression.numpy()
                    output_regression_score = output_regression_score.numpy()
                elif self.filp_fuse_type == 'type2':
                    # output_regression = (output_regression * output_regression_score + output_regression_flipped * output_regression_score_flipped)\
                    #     /(output_regression_score + output_regression_score_flipped+1e-9)
                    output_regression, output_regression_flipped = \
                        torch.from_numpy(output_regression), torch.from_numpy(output_regression_flipped)

                    output_regression_sigma, output_regression_sigma_flipped = \
                        torch.from_numpy(output_regression_sigma), torch.from_numpy(output_regression_sigma_flipped)

                    output_regression_p, output_regression_p_flipped = \
                        self.get_p(output_regression_sigma), self.get_p(output_regression_sigma_flipped)

                    p_to_coord_index = 5
                    output_regression = (
                                                    output_regression * output_regression_p ** p_to_coord_index + output_regression_flipped * output_regression_p_flipped ** p_to_coord_index) \
                                        / (
                                                    output_regression_p ** p_to_coord_index + output_regression_p_flipped ** p_to_coord_index + 1e-10)

                    output_regression_score = (output_regression_p + output_regression_p_flipped) * 0.5

                    output_regression = output_regression.numpy()
                    output_regression_score = output_regression_score.numpy()
                else:
                    NotImplementedError

        if self.with_keypoint:
            keypoint_result = self.keypoint_head.decode_keypoints(
                img_metas, output_regression, output_regression_score, [img_width, img_height])
            result.update(keypoint_result)

            if not return_heatmap:
                output_heatmap = None

            result['output_heatmap'] = output_heatmap

        return result

    def get_p(self, output_regression_sigma, p_x=0.2):
        output_regression_p = (1 - np.exp(-(p_x / output_regression_sigma)))
        output_regression_p = output_regression_p[:, :, 0] * output_regression_p[:, :, 1]
        output_regression_p = output_regression_p[:, :, None]
        return output_regression_p * 0.7
        # 0.2  0.7 7421
        # 0.2  0.7 7610
        # 0.17 0.7

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            Tensor: Output heatmaps.
        """
        output = self.backbone(img)
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            img_h, img_w = 256, 192
            img_metas = [{}]
            img_metas[0]['batch_input_shape'] = (img_h, img_w)
            img_metas[0]['img_shape'] = (img_h, img_w, 3)
            # output = self.keypoint_head(output, img_metas)
            output = self.keypoint_head(output)
        return output
