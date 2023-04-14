import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init

from mmpose.core.evaluation import (keypoint_pck_accuracy,
                                    keypoints_from_regression)
from mmpose.core.post_processing import fliplr_regression
from mmpose.models.builder import HEADS, build_loss

import torch
import torch.nn as nn
import torch.distributions as distributions
from easydict import EasyDict

def rle_fliplr_regression(regression,
                      regression_score,
                      flip_pairs,
                      center_mode='static',
                      center_x=0.5,
                      center_index=0,
                      shift=True):
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

    # flip
    # width_dim = 48
    # if shift:
    #     regression[:, :, 0] = - regression[:, :, 0] - 1 / (width_dim * 4)
    # else:
    #     regression[:, :, 0] = -1 / width_dim - regression[:, :, 0]

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


def nets():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2), nn.Tanh())

def nets3d():
    return nn.Sequential(nn.Linear(3, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 3), nn.Tanh())
    # return nn.Sequential(nn.Linear(3, 256), nn.LeakyReLU(), nn.Linear(256, 2), nn.Tanh())

def nett():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2))

def nett3d():
    return nn.Sequential(nn.Linear(3, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 3))
    # return nn.Sequential(nn.Linear(3, 256), nn.LeakyReLU(), nn.Linear(256, 2))


class Linear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())

        if self.norm:
            x_norm = torch.norm(x, dim=1, keepdim=True)
            y = y / x_norm

        if self.bias:
            y = y + self.linear.bias
        return y


class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.register_buffer('mask', mask)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])

    def _init(self):
        for m in self.t:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)
        for m in self.s:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)

    def forward_p(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def backward_p(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        DEVICE = x.device
        if self.prior.loc.device != DEVICE:
            self.prior.loc = self.prior.loc.to(DEVICE)
            self.prior.scale_tril = self.prior.scale_tril.to(DEVICE)
            self.prior._unbroadcasted_scale_tril = self.prior._unbroadcasted_scale_tril.to(DEVICE)
            self.prior.covariance_matrix = self.prior.covariance_matrix.to(DEVICE)
            self.prior.precision_matrix = self.prior.precision_matrix.to(DEVICE)

        z, logp = self.backward_p(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        x = self.forward_p(z)
        return x

    def forward(self, x):
        return self.log_prob(x)


@HEADS.register_module()
class RLERegressionHead(nn.Module):
    """Deeppose regression head with fully connected layers.

    paper ref: Alexander Toshev and Christian Szegedy,
    ``DeepPose: Human Pose Estimation via Deep Neural Networks.''.

    Args:
        in_channels (int): Number of input channels
        num_joints (int): Number of joints
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 num_joints,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.in_channels = in_channels
        self.num_joints = num_joints

        self.loss = build_loss(loss_keypoint)

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg

        # self.fc = nn.Linear(self.in_channels, self.num_joints * 2)
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fcs, out_channel = self._make_fc_layer()

        # self.fc_coord = Linear(self.in_channels, self.num_joints * 2)
        # self.fc_sigma = Linear(self.in_channels, self.num_joints * 2, norm=False)
        self.fc_coord = Linear(self.in_channels, self.num_joints * 3)
        self.fc_sigma = Linear(self.in_channels, self.num_joints * 3, norm=False)

        self.fc_layers = [self.fc_coord, self.fc_sigma]

        self.share_flow = True

        prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))

        prior3d = distributions.MultivariateNormal(torch.zeros(3), torch.eye(3))
        masks3d = torch.from_numpy(np.array([[0, 0, 1], [1, 1, 0]] * 3).astype(np.float32))

        self.flow2d = RealNVP(nets, nett, masks, prior)
        self.flow3d = RealNVP(nets3d, nett3d, masks3d, prior3d)


    # def _make_fc_layer(self):
    #     fc_layers = []
    #     num_deconv = len(self.fc_dim)
    #     input_channel = self.feature_channel
    #     for i in range(num_deconv):
    #         if self.fc_dim[i] > 0:
    #             fc = nn.Linear(input_channel, self.fc_dim[i])
    #             bn = nn.BatchNorm1d(self.fc_dim[i])
    #             fc_layers.append(fc)
    #             fc_layers.append(bn)
    #             fc_layers.append(nn.ReLU(inplace=True))
    #             input_channel = self.fc_dim[i]
    #         else:
    #             fc_layers.append(nn.Identity())
    #
    #     return nn.Sequential(*fc_layers), input_channel


    def forward(self, x):
        """Forward function."""
        # output = self.fc(x)
        # N, C = output.shape
        # return output.reshape([N, C // 2, 2])
        BATCH_SIZE = x.shape[0]
        out_coord = self.fc_coord(x).reshape(BATCH_SIZE, self.num_joints, 3)
        assert out_coord.shape[2] == 3

        out_sigma = self.fc_sigma(x).reshape(BATCH_SIZE, self.num_joints, -1)

        # (B, N, 3)
        pred_jts = out_coord.reshape(BATCH_SIZE, self.num_joints, 3)
        sigma = out_sigma.reshape(BATCH_SIZE, self.num_joints, -1).sigmoid() + 1e-9
        scores = 1 - sigma
        # (B, N, 1)
        scores = torch.mean(scores, dim=2, keepdim=True)

        output = EasyDict(
            pred_jts=pred_jts,
            sigma=sigma,
            maxvals=scores.float(),
        )
        return output

    def get_loss(self, output, target, target_weight):
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
        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 3 and target_weight.dim() == 3

        BATCH_SIZE = output.sigma.size(0)
        gt_uvd = target.reshape(output.pred_jts.shape)
        bar_mu = (output.pred_jts - gt_uvd) / output.sigma
        # (B, K, 1)
        log_phi = self.flow.log_prob(bar_mu.reshape(-1, 2)).reshape(BATCH_SIZE, self.num_joints, 1)
        output.nf_loss = torch.log(output.sigma) - log_phi
        losses['reg_loss'] = self.loss(output, target, target_weight)

        return losses

    def get_accuracy(self, output, target, target_weight):
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

        N = output.pred_jts.shape[0]

        _, avg_acc, cnt = keypoint_pck_accuracy(
            output.pred_jts.detach().cpu().numpy(),
            target.detach().cpu().numpy(),
            target_weight[:, :, 0].detach().cpu().numpy() > 0,
            thr=0.05,
            normalize=np.ones((N, 2), dtype=np.float32))
        accuracy['acc_pose'] = avg_acc

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
        output = self.forward(x)

        if flip_pairs is not None:
            output_regression, output_regression_score = rle_fliplr_regression(
                output.pred_jts.detach().cpu().numpy(), output.maxvals.detach().cpu().numpy(), flip_pairs, center_x=0.0)
        else:
            output_regression = output.pred_jts.detach().cpu().numpy()
            output_regression_score = output.maxvals.detach().cpu().numpy()
        
        output_regression += 0.5
        # output = EasyDict(
        #     preds=output_regression,
        #     maxvals=output_regression_score,
        # )
        return output_regression

    def decode(self, img_metas, output, pixel_std=200.0, **kwargs):
        """Decode the keypoints from output regression.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output (np.ndarray[N, K, 2]): predicted regression vector.
            kwargs: dict contains 'img_size'.
                img_size (tuple(img_width, img_height)): input image size.
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

        preds, maxvals = keypoints_from_regression(output, c, s, kwargs['img_size'], pixel_std)
        # maxvals = output.maxvals

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * pixel_std, axis=1)
        all_boxes[:, 5] = score

        result = {}
        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        return result

    def init_weights(self):
        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
        
        
        # for m in self.flow.t:
        #     for mm in m.modules():
        #         if isinstance(mm, nn.Linear):
        #             nn.init.xavier_uniform_(mm.weight, gain=0.01)

        # for m in self.flow.s:
        #     for mm in m.modules():
        #         if isinstance(mm, nn.Linear):
        #             nn.init.xavier_uniform_(mm.weight, gain=0.01)
        # normal_init(self.fc, mean=0, std=0.01, bias=0)
