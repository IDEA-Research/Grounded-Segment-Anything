import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.layer import make_conv_layers, make_linear_layers, make_deconv_layers
from utils.human_hand_models import smpl, mano, flame
from utils.transforms import sample_joint_features, soft_argmax_3d
from config import cfg

class PositionNet(nn.Module):
    def __init__(self, feat_dim=768):
        super(PositionNet, self).__init__()
        self.joint_num = mano.joint_num
        # print(feat_dim)
        self.conv = make_conv_layers([feat_dim, self.joint_num*cfg.output_hand_teacher_hm_shape[0]], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.hand_conv = make_conv_layers([feat_dim, 256], kernel=1, stride=1, padding=0)

    def forward(self, img_feat):
        assert (cfg.output_hand_teacher_hm_shape[1], cfg.output_hand_teacher_hm_shape[2]) == (img_feat.shape[2], img_feat.shape[3])
        joint_hm = self.conv(img_feat).view(-1,self.joint_num,cfg.output_hand_teacher_hm_shape[0],cfg.output_hand_teacher_hm_shape[1],cfg.output_hand_teacher_hm_shape[2])
        joint_coord = soft_argmax_3d(joint_hm)
        # print(img_feat.shape)
        img_feat = self.hand_conv(img_feat)
        img_feat_joints = sample_joint_features(img_feat, joint_coord.detach()[:, :, :2])
        return joint_coord, img_feat_joints, joint_hm

class MultiScaleNet(nn.Module):
    def __init__(self, feat_dim=768):
        super(MultiScaleNet, self).__init__()
        self.conv = make_conv_layers([feat_dim, feat_dim//2], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_upscale = make_deconv_layers([feat_dim//2, feat_dim//4])
        self.conv_downscale = make_conv_layers([feat_dim//2, feat_dim], kernel=3, stride=2, padding=1, bnrelu_final=False)

    def forward(self, img_feat):
        img_feat = self.conv(img_feat)
        img_feat_hr = self.conv_upscale(img_feat)
        img_feat_lr = self.conv_downscale(img_feat)
        img_feats = [img_feat_hr, img_feat, img_feat_lr]
        return img_feats

class RotationNet(nn.Module):
    def __init__(self, feat_dim=768):
        super(RotationNet, self).__init__()
        self.joint_num = mano.joint_num
        # output layers
        self.hand_conv = make_linear_layers([256, 512], relu_final=False)
        self.root_pose_out = make_linear_layers([self.joint_num*(512+3), 6], relu_final=False)
        self.hand_pose_out = make_linear_layers([self.joint_num*(512+3), (mano.orig_joint_num-1)*6], relu_final=False) # without root joint
        self.shape_out = make_linear_layers([feat_dim, mano.shape_param_dim], relu_final=False)
        self.cam_out = make_linear_layers([feat_dim, 3], relu_final=False)


    def forward(self, img_feat, img_feat_joints, joint_coord_img):
        batch_size = img_feat.shape[0]

        # shape parameter
        shape_param = self.shape_out(img_feat.mean((2,3)))

        # camera parameter
        cam_param = self.cam_out(img_feat.mean((2,3)))
        
        # pose parameter
        # print(img_feat_joints.shape)
        img_feat_joints = self.hand_conv(img_feat_joints)
        feat = torch.cat((img_feat_joints, joint_coord_img),2)
        root_pose = self.root_pose_out(feat.view(batch_size,-1))
        pose_param = self.hand_pose_out(feat.view(batch_size,-1))
        
        return root_pose, pose_param, shape_param, cam_param

class FaceRegressor(nn.Module):
    def __init__(self):
        super(FaceRegressor, self).__init__()
        self.pose_out = make_linear_layers([2048,12], relu_final=False) # pose parameter
        self.shape_out = make_linear_layers([2048, flame.shape_param_dim], relu_final=False) # shape parameter
        self.expr_out = make_linear_layers([2048, flame.expr_code_dim], relu_final=False) # expression parameter
        self.cam_out = make_linear_layers([2048,3], relu_final=False) # camera parameter

    def forward(self, img_feat):
        feat = img_feat.mean((2,3))
        
        # pose parameter
        pose_param = self.pose_out(feat)
        root_pose = pose_param[:,:6]
        jaw_pose = pose_param[:,6:]

        # shape parameter
        shape_param = self.shape_out(feat)

        # expression parameter
        expr_param = self.expr_out(feat)

        # camera parameter
        cam_param = self.cam_out(feat)

        return root_pose, jaw_pose, shape_param, expr_param, cam_param

