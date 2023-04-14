import numpy as np
import torch
import torch.nn as nn
import copy
import math
import warnings
from mmcv.cnn import build_upsample_layer, Linear, bias_init_with_prob, constant_init, normal_init
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmpose.core.evaluation import (keypoint_pck_accuracy,
                                    keypoints_from_regression)
from mmpose.core.post_processing import fliplr_regression
from mmpose.models.builder import build_loss, HEADS, build_transformer
from mmpose.core.evaluation import pose_pck_accuracy
from mmpose.models.utils.transformer import inverse_sigmoid
from mmcv.cnn import Conv2d, build_activation_layer
from mmcv.cnn.bricks.transformer import Linear, FFN, build_positional_encoding
from mmcv.cnn import ConvModule
import torch.distributions as distributions
from .rle_regression_head import nets, nett, RealNVP, nets3d, nett3d
from easydict import EasyDict
from mmpose.models.losses.regression_loss import L1Loss
from mmpose.models.losses.rle_loss import RLELoss_poseur, RLEOHKMLoss
from config import cfg
from utils.human_models import smpl_x
from torch.distributions.utils import lazy_property

from torch.distributions import MultivariateNormal


def fliplr_rle_regression(regression,
                          regression_score,
                          flip_pairs,
                          center_mode='static',
                          center_x=0.5,
                          center_index=0):
    """Flip human joints horizontally.

    Note:
        batch_size: N
        num_keypoint: K
    Args:
        regression (np.ndarray([..., K, C])): Coordinates of keypoints, where K
            is the joint number and C is the dimension. Example shapes are:
            - [N, K, C]: a batch of keypoints where N is the batch size.
            - [N, T, K, C]: a batch of pose sequences, where T is the frame
                number.
        flip_pairs (list[tuple()]): Pairs of keypoints which are mirrored
            (for example, left ear -- right ear).
        center_mode (str): The mode to set the center location on the x-axis
            to flip around. Options are:
            - static: use a static x value (see center_x also)
            - root: use a root joint (see center_index also)
        center_x (float): Set the x-axis location of the flip center. Only used
            when center_mode=static.
        center_index (int): Set the index of the root joint, whose x location
            will be used as the flip center. Only used when center_mode=root.

    Returns:
        tuple: Flipped human joints.

        - regression_flipped (np.ndarray([..., K, C])): Flipped joints.
    """
    assert regression.ndim >= 2, f'Invalid pose shape {regression.shape}'

    allowed_center_mode = {'static', 'root'}
    assert center_mode in allowed_center_mode, 'Get invalid center_mode ' \
                                               f'{center_mode}, allowed choices are {allowed_center_mode}'

    if center_mode == 'static':
        x_c = center_x
    elif center_mode == 'root':
        assert regression.shape[-2] > center_index
        x_c = regression[..., center_index:center_index + 1, 0]

    regression_flipped = regression.copy()
    regression_score_flipped = regression_score.copy()

    # Swap left-right parts
    for left, right in flip_pairs:
        regression_flipped[..., left, :] = regression[..., right, :]
        regression_flipped[..., right, :] = regression[..., left, :]
        regression_score_flipped[..., left, :] = regression_score[..., right, :]
        regression_score_flipped[..., right, :] = regression_score[..., left, :]

    # Flip horizontally
    regression_flipped[..., 0] = x_c * 2 - regression_flipped[..., 0]
    return regression_flipped, regression_score_flipped


class Linear_with_norm(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear_with_norm, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())

        if self.norm:
            x_norm = torch.norm(x, dim=-1, keepdim=True)
            y = y / x_norm

        if self.bias:
            y = y + self.linear.bias
        return y

def deepapply(obj, fn):
    r"""Applies `fn` to all tensors referenced in `obj`"""

    if torch.is_tensor(obj):
        obj = fn(obj)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = deepapply(value, fn)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = deepapply(value, fn)
    elif isinstance(obj, tuple):
        obj = tuple(
            deepapply(value, fn)
            for value in obj
        )
    elif hasattr(obj, '__dict__'):
        deepapply(obj.__dict__, fn)

    return obj


__init__ = MultivariateNormal.__init__


def init(self, *args, **kwargs):
    __init__(self, *args, **kwargs)

    self.__class__ = type(
        self.__class__.__name__,
        (self.__class__, nn.Module),
        {},
    )

    nn.Module.__init__(self)


MultivariateNormal.__init__ = init
MultivariateNormal._apply = deepapply


@HEADS.register_module()
class Poseur_noise_sample(nn.Module):
    """
    rle loss for transformer_utils
    """

    def __init__(self,
                 in_channels,
                 num_queries=17,
                 num_reg_fcs=2,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 transformer=None,
                 with_box_refine=False,
                 as_two_stage=False,
                 heatmap_size=[64, 48],
                 num_joints=17,
                 loss_coord_enc=None,
                 loss_coord_dec=None,
                 loss_hp_keypoint=None,
                 use_heatmap_loss=True,
                 train_cfg=None,
                 test_cfg=None,
                 use_udp=False,
                 ):
        super().__init__()
        self.use_udp = use_udp
        self.num_queries = num_queries
        self.num_reg_fcs = num_reg_fcs
        self.in_channels = in_channels
        self.act_cfg = transformer.get('act_cfg', dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
                                                 f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
                                                 f' and {num_feats}.'

        self.num_joints = num_joints
        # self.num_joints = len(smpl_x.pos_joint_part['rhand'])
        self.heatmap_size = heatmap_size
        self.loss_coord_enc = build_loss(loss_coord_enc)
        self.loss_coord_dec = build_loss(loss_coord_dec)

        self.use_dec_rle_loss = isinstance(self.loss_coord_dec, RLELoss_poseur) or isinstance(self.loss_coord_dec,
                                                                                              RLEOHKMLoss)
        self.use_heatmap_loss = use_heatmap_loss
        if self.use_heatmap_loss:
            self.loss_hp = build_loss(loss_hp_keypoint)

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg

        enc_prior = MultivariateNormal(torch.zeros(2), torch.eye(2))
        dec_prior = MultivariateNormal(torch.zeros(2), torch.eye(2))
        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))

        enc_prior3d = MultivariateNormal(torch.zeros(3), torch.eye(3))
        dec_prior3d = MultivariateNormal(torch.zeros(3), torch.eye(3))
        masks3d = torch.from_numpy(np.array([[0, 0, 1], [1, 1, 0]] * 3).astype(np.float32))

        self.enc_flow2d = RealNVP(nets, nett, masks, enc_prior)
        self.enc_flow3d = RealNVP(nets3d, nett3d, masks3d, enc_prior3d)

        if self.use_dec_rle_loss:
            self.dec_flow2d = RealNVP(nets, nett, masks, dec_prior)
            self.dec_flow3d = RealNVP(nets3d, nett3d, masks3d, dec_prior3d)

        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        fc_coord_branch = []
        for _ in range(self.num_reg_fcs):
            fc_coord_branch.append(Linear(self.embed_dims, self.embed_dims))
            fc_coord_branch.append(nn.ReLU())
        fc_coord_branch.append(Linear(self.embed_dims, 3))
        fc_coord_branch = nn.Sequential(*fc_coord_branch)

        if self.use_dec_rle_loss:
            fc_sigma_branch = []
            for _ in range(self.num_reg_fcs):
                fc_sigma_branch.append(Linear(self.embed_dims, self.embed_dims))
            fc_sigma_branch.append(Linear_with_norm(self.embed_dims, 3, norm=False))
            fc_sigma_branch = nn.Sequential(*fc_sigma_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.fc_coord_branches = _get_clones(fc_coord_branch, num_pred)
            self.fc_coord_output_branches = _get_clones(fc_coord_branch, num_pred)
            if self.use_dec_rle_loss:
                self.fc_sigma_branches = _get_clones(fc_sigma_branch, num_pred)
        else:
            self.fc_coord_branches = nn.ModuleList(
                [fc_coord_branch for _ in range(num_pred)])
            if isinstance(self.loss_coord_dec, RLELoss) or isinstance(self.loss_coord_dec, RLEOHKMLoss):
                self.fc_sigma_branches = nn.ModuleList([fc_sigma_branch for _ in range(1)])

        if self.as_two_stage:
            self.query_embedding = None
        else:
            self.query_embedding = nn.Embedding(self.num_queries,
                                                self.embed_dims * 2)

        if self.use_heatmap_loss:
            from mmcv.cnn import build_upsample_layer
            # simplebaseline style
            num_layers = 3
            num_kernels = [4, 4, 4]
            num_filters = [256, 256, 256]

            layers = []
            for i in range(num_layers):
                kernel, padding, output_padding = \
                    self._get_deconv_cfg(num_kernels[i])

                planes = num_filters[i]
                if i == 0:
                    layers.append(
                        build_upsample_layer(
                            dict(type='deconv'),
                            in_channels=self.embed_dims,
                            out_channels=planes,
                            kernel_size=kernel,
                            stride=2,
                            padding=padding,
                            output_padding=output_padding,
                            bias=False))
                else:
                    layers.append(
                        build_upsample_layer(
                            dict(type='deconv'),
                            in_channels=planes,
                            out_channels=planes,
                            kernel_size=kernel,
                            stride=2,
                            padding=padding,
                            output_padding=output_padding,
                            bias=False))

                layers.append(nn.BatchNorm2d(planes))
                layers.append(nn.ReLU(inplace=True))
                self.in_channels = planes

            self.deconv_layer = nn.Sequential(*layers)
            self.final_layer = nn.Sequential(
                ConvModule(
                    self.embed_dims,
                    self.num_joints,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_cfg=None,
                    act_cfg=None,
                    inplace=False)
            )

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()

        # for m in [self.fc_coord_branches, self.fc_sigma_branches]:
        for m in [self.fc_coord_branches]:
            for mm in m:
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)

        for m in [self.fc_coord_output_branches]:
            for mm in m:
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)

        if self.use_heatmap_loss:
            for _, m in self.deconv_layer.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001, bias=0)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)

    def forward(self, mlvl_feats, coord_init=None, query_init=None):

        batch_size = mlvl_feats[0].size(0)
        img_w, img_h = self.train_cfg['image_size']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, img_h, img_w))
        for img_id in range(batch_size):
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight

        memory, spatial_shapes, level_start_index, hs, init_reference, inter_references, \
        enc_outputs = self.transformer(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            reg_branches=self.fc_coord_branches if self.with_box_refine else None,  # noqa:E501
            cls_branches=None,  # noqa:E501
            coord_init=coord_init,
            query_init=query_init,
        )
        hs = hs.permute(0, 2, 1, 3)
        outputs_coords = []

        dec_outputs = EasyDict(pred_jts=outputs_coords, feat=hs)

        return enc_outputs, dec_outputs

    def get_loss(self, enc_output, dec_output, coord_target, coord_target_weight, hp_target, hp_target_weight):
        losses = dict()
        if self.as_two_stage and enc_output is not None:
            enc_rle_loss = self.get_enc_rle_loss(enc_output, coord_target, coord_target_weight)
            losses.update(enc_rle_loss)

        dec_rle_loss = self.get_dec_rle_loss(dec_output, coord_target, coord_target_weight)
        losses.update(dec_rle_loss)

        return losses

    def get_enc_rle_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.
        Note:
            batch_size: N
            num_keypoints: K
        Args:
            output (torch.Tensor[N, K, 2]): Output keypoints.
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """

        losses = dict()
        assert not isinstance(self.loss_coord_enc, nn.Sequential)
        assert target.dim() == 3 and target_weight.dim() == 3

        BATCH_SIZE = output.sigma.size(0)
        gt_uvd = target.reshape(output.pred_jts.shape)
        gt_uvd_weight = target_weight.reshape(output.pred_jts.shape)
        gt_3d_mask = gt_uvd_weight[:, :, 2].reshape(-1)

        assert output.pred_jts.shape == output.sigma.shape, (output.pred_jts.shape, output.sigma.shape)
        bar_mu = (output.pred_jts - gt_uvd) / output.sigma
        bar_mu = bar_mu.reshape(-1, 3)
        bar_mu_3d = bar_mu[gt_3d_mask > 0]
        bar_mu_2d = bar_mu[gt_3d_mask < 1][:, :2]
        # (B, K, 3)
        log_phi_3d = self.enc_flow3d.log_prob(bar_mu_3d)
        log_phi_2d = self.enc_flow2d.log_prob(bar_mu_2d)
        log_phi = torch.zeros_like(bar_mu[:, 0])
        # print(gt_3d_mask)
        log_phi[gt_3d_mask > 0] = log_phi_3d
        log_phi[gt_3d_mask < 1] = log_phi_2d
        log_phi = log_phi.reshape(BATCH_SIZE, self.num_joints, 1)

        output.nf_loss = torch.log(output.sigma) - log_phi
        losses['enc_rle_loss'] = self.loss_coord_enc(output, target, target_weight)

        return losses

    def get_enc_rle_loss_old(self, output, target, target_weight):
        """Calculate top-down keypoint loss.
        Note:
            batch_size: N
            num_keypoints: K
        Args:
            output (torch.Tensor[N, K, 2]): Output keypoints.
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """

        losses = dict()
        assert not isinstance(self.loss_coord_enc, nn.Sequential)
        assert target.dim() == 3 and target_weight.dim() == 3

        BATCH_SIZE = output.sigma.size(0)
        gt_uv = target.reshape(output.pred_jts.shape)
        bar_mu = (output.pred_jts - gt_uv) / output.sigma
        # (B, K, 1)
        log_phi = self.enc_flow.log_prob(bar_mu.reshape(-1, 2)).reshape(BATCH_SIZE, self.num_joints, 1)
        output.nf_loss = torch.log(output.sigma) - log_phi
        losses['enc_rle_loss'] = self.loss_coord_enc(output, target, target_weight)

        return losses

    def get_dec_rle_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2]): Output keypoints.
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """

        losses = dict()
        assert not isinstance(self.loss_coord_dec, nn.Sequential)
        assert target.dim() == 3 and target_weight.dim() == 3
        target = target.repeat(1, self.transformer.num_noise_sample + 1, 1)
        target_weight = target_weight.repeat(1, self.transformer.num_noise_sample + 1, 1)

        if self.with_box_refine:
            if self.use_dec_rle_loss:
                for i in range(len(output.pred_jts)):
                    pred_jts, sigma = output.pred_jts[i], output.sigma[i]
                    output_i = EasyDict(
                        pred_jts=pred_jts,
                        sigma=sigma
                    )
                    BATCH_SIZE = output_i.sigma.size(0)
                    gt_uvd = target.reshape(output_i.pred_jts.shape)
                    gt_uvd_weight = target_weight.reshape(pred_jts.shape)
                    gt_3d_mask = gt_uvd_weight[:, :, 2].reshape(-1)

                    assert pred_jts.shape == sigma.shape, (pred_jts.shape, sigma.shape)
                    bar_mu = (output_i.pred_jts - gt_uvd) / output_i.sigma
                    bar_mu = bar_mu.reshape(-1, 3)
                    bar_mu_3d = bar_mu[gt_3d_mask > 0]
                    bar_mu_2d = bar_mu[gt_3d_mask < 1][:, :2]
                    # (B, K, 3)
                    log_phi_3d = self.dec_flow3d.log_prob(bar_mu_3d)
                    log_phi_2d = self.dec_flow2d.log_prob(bar_mu_2d)
                    log_phi = torch.zeros_like(bar_mu[:, 0])
                    log_phi[gt_3d_mask > 0] = log_phi_3d
                    log_phi[gt_3d_mask < 1] = log_phi_2d
                    log_phi = log_phi.reshape(BATCH_SIZE, self.num_joints * (self.transformer.num_noise_sample + 1), 1)
                    output_i.nf_loss = torch.log(output_i.sigma) - log_phi
                    losses['dec_rle_loss_{}'.format(i)] = self.loss_coord_dec(output_i, target, target_weight)
            else:
                for i, pred_jts in enumerate(output.pred_jts):
                    losses['dec_rle_loss_{}'.format(i)] = self.loss_coord_dec(pred_jts, target, target_weight)
        else:
            if self.use_dec_rle_loss:
                BATCH_SIZE = output.sigma.size(0)
                gt_uv = target.reshape(output.pred_jts.shape)
                bar_mu = (output.pred_jts - gt_uv) / output.sigma
                # (B, K, 1)
                log_phi = self.dec_flow.log_prob(bar_mu.reshape(-1, 2)).reshape(BATCH_SIZE, self.num_joints, 1)
                output.nf_loss = torch.log(output.sigma) - log_phi
                losses['dec_rle_loss'] = self.loss_coord_dec(output, target, target_weight) * 0
            else:
                losses['dec_rle_loss'] = self.loss_coord_dec(output.pred_jts, target + 0.5, target_weight) * 0

        return losses

    def get_hp_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K
            heatmaps height: H
            heatmaps weight: W

        Args:
            output (torch.Tensor[NxKxHxW]): Output heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]):
                Weights across different joint types.
        """

        losses = dict()

        if isinstance(self.loss_hp, nn.Sequential):
            if not isinstance(output, dict):
                assert len(self.loss_hp) == output.size(0)
                assert target.dim() == 5 and target_weight.dim() == 4
                num_hp_layers = output.size(0)
                for i in range(num_hp_layers):
                    target_i = target[:, i, :, :, :]
                    target_weight_i = target_weight[:, i, :, :]
                    losses['mse_loss_{}'.format(i)] = self.loss_hp[i](output[i], target_i, target_weight_i)
            else:
                out_hp_backbone = output['backbone']
                num_hp_layers = out_hp_backbone.size(0)
                for i in range(num_hp_layers):
                    target_i = target[:, i, :, :, :]
                    target_weight_i = target_weight[:, i, :, :]
                    losses['mse_loss_backbone_{}'.format(i)] = self.loss_hp[i](out_hp_backbone[i], target_i,
                                                                               target_weight_i)

                out_hp_enc = output['enc']
                for lvl in range(len(out_hp_enc)):
                    if lvl == 2 or lvl == 5:
                        # if lvl == 5:
                        for i in range(3):
                            target_i = target[:, i + 1, :, :, :]
                            target_weight_i = target_weight[:, i + 1, :, :]
                            # losses['reg_loss'] += self.loss(output[i], target, target_weight).sum()
                            if lvl == 2:
                                loss_weight = 0.1
                            elif lvl == 5:
                                loss_weight = 1.0

                            losses['mse_loss_enc_layer{}_c{}'.format(lvl, i + 3)] = loss_weight * self.loss_hp[i + 1](
                                out_hp_enc[lvl][i], target_i, target_weight_i)
        else:

            assert target.dim() == 4 and target_weight.dim() == 3
            losses['mse_loss'] = self.loss_hp(output, target, target_weight)

        return losses

    def get_accuracy(self, enc_output, dec_output, coord_target, coord_target_weight, hp_target, hp_target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2]): Output keypoints.
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """

        accuracy = dict()
        # coord_output = output["coord"]
        if self.as_two_stage and enc_output is not None:
            coord_output = enc_output.pred_jts
            N = coord_output.shape[0]

            _, avg_acc, cnt = keypoint_pck_accuracy(
                coord_output.detach().cpu().numpy(),
                coord_target.detach().cpu().numpy(),
                coord_target_weight[:, :, 0].detach().cpu().numpy() > 0,
                thr=0.05,
                normalize=np.ones((N, 2), dtype=np.float32))
            accuracy['enc_coord_acc'] = avg_acc

        coord_output = dec_output.pred_jts
        if coord_output.dim() == 4:
            coord_output = coord_output[-1]
        N = coord_output.shape[0]

        if not self.use_dec_rle_loss:
            coord_target += 0.5
        # self.num_joints
        _, avg_acc, cnt = keypoint_pck_accuracy(
            coord_output[:, :self.num_joints].detach().cpu().numpy(),
            coord_target.detach().cpu().numpy(),
            coord_target_weight[:, :, 0].detach().cpu().numpy() > 0,
            thr=0.05,
            normalize=np.ones((N, 2), dtype=np.float32))
        accuracy['dec_coord_acc'] = avg_acc

        # if self.use_heatmap_loss and self.use_multi_stage_memory:
        #     assert hp_target.dim() == 5 and hp_target_weight.dim() == 4
        #     _, avg_acc, _ = pose_pck_accuracy(
        #         hp_output_backbone[0].detach().cpu().numpy(),
        #         hp_target[:, 0, ...].detach().cpu().numpy(),
        #         hp_target_weight[:, 0,
        #                       ...].detach().cpu().numpy().squeeze(-1) > 0)
        #     accuracy['hp_acc_backbone'] = float(avg_acc)

        #     _, avg_acc, _ = pose_pck_accuracy(
        #         hp_output_enc[-1][0].detach().cpu().numpy(),
        #         hp_target[:, 1, ...].detach().cpu().numpy(),
        #         hp_target_weight[:, 1,
        #                       ...].detach().cpu().numpy().squeeze(-1) > 0)
        #     accuracy['hp_acc_enc'] = float(avg_acc)

        # else:
        if self.use_heatmap_loss:
            hp_output = dec_output["hp"]
            _, avg_acc, _ = pose_pck_accuracy(
                hp_output.detach().cpu().numpy(),
                hp_target.detach().cpu().numpy(),
                hp_target_weight.detach().cpu().numpy().squeeze(-1) > 0)
            accuracy['hp_acc'] = float(avg_acc)

        return accuracy

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_regression (np.ndarray): Output regression.

        Args:
            x (torch.Tensor[N, K, 2]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """
        output_enc, output_dec = self.forward(x)
        output_regression, output_regression_score = output_dec.pred_jts.detach().cpu().numpy(), output_dec.maxvals.detach().cpu().numpy()
        output_sigma = output_dec.sigma.detach().cpu().numpy()
        output_sigma = output_sigma[-1]
        output_regression_score = np.concatenate([output_regression_score, output_sigma], axis=2)

        if output_regression.ndim == 4:
            output_regression = output_regression[-1]

        if flip_pairs is not None:

            output_regression, output_regression_score = fliplr_rle_regression(
                output_regression, output_regression_score, flip_pairs)

        return output_regression, output_regression_score

    def decode_keypoints(self, img_metas, output_regression, output_regression_score, img_size):
        """Decode keypoints from output regression.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output_regression (np.ndarray[N, K, 2]): model
                predicted regression vector.
            img_size (tuple(img_width, img_height)): model input image size.
        """
        batch_size = len(img_metas)

        if 'bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['center']
            s[i, :] = img_metas[i]['scale']
            image_paths.append(img_metas[i]['image_file'])

            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)

            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])

        preds, maxvals = keypoints_from_regression(output_regression, c, s,
                                                   img_size)

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        # all_preds[:, :, 2:3] = maxvals
        all_preds[:, :, 2:3] = output_regression_score
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {}

        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        return result
