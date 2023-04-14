import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from nets.layer import make_conv_layers, make_linear_layers, make_deconv_layers
from utils.transforms import sample_joint_features, soft_argmax_2d, soft_argmax_3d
from utils.human_models import smpl_x
from config import cfg
import math


class PositionNet(nn.Module):
    def __init__(self, part, resnet_type):
        super(PositionNet, self).__init__()
        if part == 'body':
            self.joint_num = len(smpl_x.pos_joint_part['body'])
            self.hm_shape = cfg.output_hm_shape
        elif part == 'hand':
            self.joint_num = len(smpl_x.pos_joint_part['rhand'])
            self.hm_shape = cfg.output_hand_hm_shape
        if resnet_type == 18:
            feat_dim = 512
        elif resnet_type == 50:
            feat_dim = 2048
        self.conv = make_conv_layers([feat_dim, self.joint_num * self.hm_shape[0]], kernel=1, stride=1, padding=0,
                                     bnrelu_final=False)

    def forward(self, img_feat):
        joint_hm = self.conv(img_feat).view(-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2])
        joint_coord = soft_argmax_3d(joint_hm)
        joint_hm = F.softmax(joint_hm.view(-1, self.joint_num, self.hm_shape[0] * self.hm_shape[1] * self.hm_shape[2]),
                             2)
        joint_hm = joint_hm.view(-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2])
        return joint_hm, joint_coord


class RotationNet(nn.Module):
    def __init__(self, part, resnet_type):
        super(RotationNet, self).__init__()
        self.part = part
        if part == 'body':
            self.joint_num = len(smpl_x.pos_joint_part['body']) + 4 + 4  # body + lhand MCP joints + rhand MCP joints
        elif part == 'hand':
            self.joint_num = len(smpl_x.pos_joint_part['rhand'])
        if resnet_type == 18:
            feat_dim = 512
        elif resnet_type == 50:
            feat_dim = 2048

        if part == 'body':
            self.body_conv = make_conv_layers([feat_dim, 512], kernel=1, stride=1, padding=0)
            self.lhand_conv = make_conv_layers([feat_dim, 512], kernel=1, stride=1, padding=0)
            self.rhand_conv = make_conv_layers([feat_dim, 512], kernel=1, stride=1, padding=0)
            self.root_pose_out = make_linear_layers([self.joint_num * 515, 6], relu_final=False)
            self.body_pose_out = make_linear_layers(
                [self.joint_num * 515, (len(smpl_x.orig_joint_part['body']) - 1) * 6], relu_final=False)  # without root
            self.shape_out = make_linear_layers([feat_dim, smpl_x.shape_param_dim], relu_final=False)
            self.cam_out = make_linear_layers([feat_dim, 3], relu_final=False)
        elif part == 'hand':
            self.hand_conv = make_conv_layers([feat_dim, 512], kernel=1, stride=1, padding=0)
            self.hand_pose_out = make_linear_layers([self.joint_num * 515, len(smpl_x.orig_joint_part['rhand']) * 6],
                                                    relu_final=False)

    def forward(self, img_feat, joint_coord_img, lhand_img_feat=None, lhand_joint_coord_img=None, rhand_img_feat=None,
                rhand_joint_coord_img=None):
        batch_size = img_feat.shape[0]

        if self.part == 'body':
            # shape parameter
            shape_param = self.shape_out(img_feat.mean((2, 3)))

            # camera parameter
            cam_param = self.cam_out(img_feat.mean((2, 3)))

            # body pose parameter
            # body feature
            body_img_feat = self.body_conv(img_feat)
            body_img_feat = sample_joint_features(body_img_feat, joint_coord_img[:, :, :2])
            body_feat = torch.cat((body_img_feat, joint_coord_img), 2)  # batch_size, joint_num (body), 512+3
            # lhand feature
            lhand_img_feat = self.lhand_conv(lhand_img_feat)
            lhand_img_feat = sample_joint_features(lhand_img_feat, lhand_joint_coord_img[:, :, :2])
            lhand_feat = torch.cat((lhand_img_feat, lhand_joint_coord_img), 2)  # batch_size, joint_num (4), 512+3
            # rhand feature
            rhand_img_feat = self.rhand_conv(rhand_img_feat)
            rhand_img_feat = sample_joint_features(rhand_img_feat, rhand_joint_coord_img[:, :, :2])
            rhand_feat = torch.cat((rhand_img_feat, rhand_joint_coord_img), 2)  # batch_size, joint_num (4), 512+3
            # forward to fc
            feat = torch.cat((body_feat, lhand_feat, rhand_feat), 1)
            root_pose = self.root_pose_out(feat.view(batch_size, -1))
            body_pose = self.body_pose_out(feat.view(batch_size, -1))
            return root_pose, body_pose, shape_param, cam_param

        elif self.part == 'hand':
            # hand pose parameter
            img_feat = self.hand_conv(img_feat)
            img_feat_joints = sample_joint_features(img_feat, joint_coord_img[:, :, :2])
            feat = torch.cat((img_feat_joints, joint_coord_img), 2)  # batch_size, joint_num, 512+3
            hand_pose = self.hand_pose_out(feat.view(batch_size, -1))
            return hand_pose


class FaceRegressor(nn.Module):
    def __init__(self):
        super(FaceRegressor, self).__init__()
        self.expr_out = make_linear_layers([512, smpl_x.expr_code_dim], relu_final=False)
        self.jaw_pose_out = make_linear_layers([512, 6], relu_final=False)

    def forward(self, img_feat):
        expr_param = self.expr_out(img_feat.mean((2, 3)))  # expression parameter
        jaw_pose = self.jaw_pose_out(img_feat.mean((2, 3)))  # jaw pose parameter
        return expr_param, jaw_pose


class BoxNet(nn.Module):
    def __init__(self):
        super(BoxNet, self).__init__()
        self.joint_num = len(smpl_x.pos_joint_part['body'])
        self.deconv = make_deconv_layers([2048 + self.joint_num * cfg.output_hm_shape[0], 256, 256, 256])
        self.bbox_center = make_conv_layers([256, 3], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.lhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.rhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.face_size = make_linear_layers([256, 256, 2], relu_final=False)

    def forward(self, img_feat, joint_hm, joint_img):
        joint_hm = joint_hm.view(joint_hm.shape[0], joint_hm.shape[1] * cfg.output_hm_shape[0], cfg.output_hm_shape[1],
                                 cfg.output_hm_shape[2])
        img_feat = torch.cat((img_feat, joint_hm), 1)
        img_feat = self.deconv(img_feat)

        # bbox center
        bbox_center_hm = self.bbox_center(img_feat)
        bbox_center = soft_argmax_2d(bbox_center_hm)
        lhand_center, rhand_center, face_center = bbox_center[:, 0, :], bbox_center[:, 1, :], bbox_center[:, 2, :]

        # bbox size
        lhand_feat = sample_joint_features(img_feat, lhand_center[:, None, :].detach())[:, 0, :]
        lhand_size = self.lhand_size(lhand_feat)
        rhand_feat = sample_joint_features(img_feat, rhand_center[:, None, :].detach())[:, 0, :]
        rhand_size = self.rhand_size(rhand_feat)
        face_feat = sample_joint_features(img_feat, face_center[:, None, :].detach())[:, 0, :]
        face_size = self.face_size(face_feat)

        lhand_center = lhand_center / 8
        rhand_center = rhand_center / 8
        face_center = face_center / 8
        return lhand_center, lhand_size, rhand_center, rhand_size, face_center, face_size


class HandRoI(nn.Module):
    def __init__(self, backbone):
        super(HandRoI, self).__init__()
        self.backbone = backbone

    def forward(self, img, lhand_bbox, rhand_bbox):
        lhand_bbox = torch.cat((torch.arange(lhand_bbox.shape[0]).float().cuda()[:, None], lhand_bbox),
                               1)  # batch_idx, xmin, ymin, xmax, ymax
        rhand_bbox = torch.cat((torch.arange(rhand_bbox.shape[0]).float().cuda()[:, None], rhand_bbox),
                               1)  # batch_idx, xmin, ymin, xmax, ymax

        lhand_bbox_roi = lhand_bbox.clone()
        lhand_bbox_roi[:, 1] = lhand_bbox_roi[:, 1] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        lhand_bbox_roi[:, 2] = lhand_bbox_roi[:, 2] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        lhand_bbox_roi[:, 3] = lhand_bbox_roi[:, 3] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        lhand_bbox_roi[:, 4] = lhand_bbox_roi[:, 4] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        lhand_img = torchvision.ops.roi_align(img, lhand_bbox_roi, cfg.input_hand_shape, aligned=False)
        lhand_img = torch.flip(lhand_img, [3])  # flip to the right hand

        rhand_bbox_roi = rhand_bbox.clone()
        rhand_bbox_roi[:, 1] = rhand_bbox_roi[:, 1] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        rhand_bbox_roi[:, 2] = rhand_bbox_roi[:, 2] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        rhand_bbox_roi[:, 3] = rhand_bbox_roi[:, 3] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        rhand_bbox_roi[:, 4] = rhand_bbox_roi[:, 4] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        rhand_img = torchvision.ops.roi_align(img, rhand_bbox_roi, cfg.input_hand_shape, aligned=False)

        hand_img = torch.cat((lhand_img, rhand_img))
        hand_feat = self.backbone(hand_img)
        return hand_feat


class FaceRoI(nn.Module):
    def __init__(self, backbone):
        super(FaceRoI, self).__init__()
        self.backbone = backbone

    def forward(self, img, face_bbox):
        face_bbox = torch.cat((torch.arange(face_bbox.shape[0]).float().cuda()[:, None], face_bbox),
                              1)  # batch_idx, xmin, ymin, xmax, ymax

        face_bbox_roi = face_bbox.clone()
        face_bbox_roi[:, 1] = face_bbox_roi[:, 1] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        face_bbox_roi[:, 2] = face_bbox_roi[:, 2] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        face_bbox_roi[:, 3] = face_bbox_roi[:, 3] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        face_bbox_roi[:, 4] = face_bbox_roi[:, 4] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        face_img = torchvision.ops.roi_align(img, face_bbox_roi, cfg.input_face_shape, aligned=False)

        face_feat = self.backbone(face_img)
        return face_feat