import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from nets.layer import make_conv_layers, make_linear_layers, make_deconv_layers
from utils.transforms import sample_joint_features, soft_argmax_2d, soft_argmax_3d
from utils.human_models import smpl_x
from config import cfg
import math
from mmcv.ops.roi_align import roi_align
import cv2
import os

class PositionNet(nn.Module):
    def __init__(self, part, feat_dim=768):
        super(PositionNet, self).__init__()
        if part == 'body':
            self.joint_num = len(smpl_x.pos_joint_part['body'])
            self.hm_shape = cfg.output_hm_shape
        elif part == 'hand':
            self.joint_num = cfg.hand_pos_joint_num
            self.hm_shape = cfg.output_hand_hm_shape
            self.hand_conv = make_conv_layers([feat_dim, 256], kernel=1, stride=1, padding=0)

        self.conv = make_conv_layers([feat_dim, self.joint_num * self.hm_shape[0]], kernel=1, stride=1, padding=0,
                                     bnrelu_final=False)
        self.part = part

    def forward(self, img_feat):
        assert (img_feat.shape[2], img_feat.shape[3]) == (self.hm_shape[1], self.hm_shape[2])
        joint_hm = self.conv(img_feat).view(-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2])
        joint_coord = soft_argmax_3d(joint_hm)
        joint_hm = F.softmax(joint_hm.view(-1, self.joint_num, self.hm_shape[0] * self.hm_shape[1] * self.hm_shape[2]),
                             2)
        joint_hm = joint_hm.view(-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2])
        if self.part=='hand':
            img_feat = self.hand_conv(img_feat)
            img_feat_joints = sample_joint_features(img_feat, joint_coord.detach()[:, :, :2])
            return joint_hm, joint_coord, img_feat_joints
        return joint_hm, joint_coord


class RotationNet(nn.Module):
    def __init__(self, part, feat_dim = 768):
        super(RotationNet, self).__init__()
        self.part = part
        self.joint_num = cfg.hand_pos_joint_num

        self.hand_conv = make_linear_layers([feat_dim, 512], relu_final=False)
        self.hand_pose_out = make_linear_layers([self.joint_num * 515, len(smpl_x.orig_joint_part['rhand']) * 6],
                                                    relu_final=False)
        self.feat_dim = feat_dim

    def forward(self, img_feat_joints, joint_coord_img):
        batch_size = img_feat_joints.shape[0]
        # hand pose parameter
        img_feat_joints = self.hand_conv(img_feat_joints)
        feat = torch.cat((img_feat_joints, joint_coord_img), 2)  # batch_size, joint_num, 512+3
        hand_pose = self.hand_pose_out(feat.view(batch_size, -1))
        return hand_pose


class BodyRotationNet(nn.Module):
    def __init__(self, feat_dim = 768):
        super(BodyRotationNet, self).__init__()
        self.joint_num = len(smpl_x.pos_joint_part['body']) + 4 + 4
        self.body_conv = make_linear_layers([feat_dim, 512], relu_final=False)
        self.root_pose_out = make_linear_layers([self.joint_num * (512+3), 6], relu_final=False)
        self.body_pose_out = make_linear_layers(
            [self.joint_num * (512+3), (len(smpl_x.orig_joint_part['body']) - 1) * 6], relu_final=False)  # without root
        self.shape_out = make_linear_layers([feat_dim, smpl_x.shape_param_dim], relu_final=False)
        self.cam_out = make_linear_layers([feat_dim, 3], relu_final=False)
        self.feat_dim = feat_dim
        self.lhand_conv = make_conv_layers([2048, 512], kernel=1, stride=1, padding=0)
        self.rhand_conv = make_conv_layers([2048, 512], kernel=1, stride=1, padding=0)

    def forward(self, body_pose_token, shape_token, cam_token, body_joint_img, lhand_img_feat=None,
                lhand_joint_coord_img=None, rhand_img_feat=None, rhand_joint_coord_img=None):
        '''
        body_pose_token: [bs,N_J,c]
        shape_token: [bs,c]
        cam_token: [bs,c]
        body_joint_img: [bs,N_J,3]
        :return: [bs,c]
        '''
        # print(shape_token.shape)
        batch_size = body_pose_token.shape[0]

        # shape parameter
        shape_param = self.shape_out(shape_token)

        # camera parameter
        cam_param = self.cam_out(cam_token)

        # body pose parameter
        # body feature
        body_pose_token = self.body_conv(body_pose_token)
        body_feat = torch.cat((body_pose_token, body_joint_img), 2)
        # lhand feature
        # print(lhand_img_feat.shape)
        lhand_img_feat = self.lhand_conv(lhand_img_feat)
        lhand_img_feat = sample_joint_features(lhand_img_feat, lhand_joint_coord_img[:, :, :2])
        lhand_feat = torch.cat((lhand_img_feat, lhand_joint_coord_img), dim=2)
        # rhand feature
        rhand_img_feat = self.rhand_conv(rhand_img_feat)
        rhand_img_feat = sample_joint_features(rhand_img_feat, rhand_joint_coord_img[:, :, :2])
        rhand_feat = torch.cat((rhand_img_feat, rhand_joint_coord_img), dim=2)
        # forward to fc
        feat = torch.cat((body_feat, lhand_feat, rhand_feat), 1)
        root_pose = self.root_pose_out(feat.view(batch_size, -1))
        body_pose = self.body_pose_out(feat.view(batch_size, -1))

        return root_pose, body_pose, shape_param, cam_param

class FaceRegressor(nn.Module):
    def __init__(self, feat_dim=768):
        super(FaceRegressor, self).__init__()
        self.expr_out = make_linear_layers([feat_dim, smpl_x.expr_code_dim], relu_final=False)
        self.jaw_pose_out = make_linear_layers([feat_dim, 6], relu_final=False)

    def forward(self, expr_token, jaw_pose_token):
        expr_param = self.expr_out(expr_token)  # expression parameter
        jaw_pose = self.jaw_pose_out(jaw_pose_token)  # jaw pose parameter
        return expr_param, jaw_pose

# class FaceRegressor(nn.Module):
#     def __init__(self):
#         super(FaceRegressor, self).__init__()
#         self.expr_out = make_linear_layers([512, smpl_x.expr_code_dim], relu_final=False)
#         self.jaw_pose_out = make_linear_layers([512, 6], relu_final=False)
#
#     def forward(self, img_feat):
#         expr_param = self.expr_out(img_feat.mean((2, 3)))  # expression parameter
#         jaw_pose = self.jaw_pose_out(img_feat.mean((2, 3)))  # jaw pose parameter
#         return expr_param, jaw_pose


class BoxNet(nn.Module):
    def __init__(self, feat_dim=768):
        super(BoxNet, self).__init__()
        self.joint_num = len(smpl_x.pos_joint_part['body'])
        self.deconv = make_deconv_layers([feat_dim + self.joint_num * cfg.output_hm_shape[0], 256, 256, 256])
        self.bbox_center = make_conv_layers([256, 3], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.lhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.rhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.face_size = make_linear_layers([256, 256, 2], relu_final=False)

    def forward(self, img_feat, joint_hm, joint_img=None):
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


class BoxSizeNet(nn.Module):
    def __init__(self):
        super(BoxSizeNet, self).__init__()
        self.lhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.rhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.face_size = make_linear_layers([256, 256, 2], relu_final=False)

    def forward(self, box_fea):
        # box_fea: [bs, 3, C]
        lhand_size = self.lhand_size(box_fea[:, 0])
        rhand_size = self.rhand_size(box_fea[:, 1])
        face_size = self.face_size(box_fea[:, 2])
        return lhand_size, rhand_size, face_size


class HandRoI(nn.Module):
    def __init__(self, feat_dim=768, upscale=4):
        super(HandRoI, self).__init__()
        self.deconv = nn.ModuleList([])
        for i in range(int(math.log2(upscale))+1):
            if i==0:
                self.deconv.append(make_conv_layers([feat_dim, feat_dim], kernel=1, stride=1, padding=0, bnrelu_final=False))
            elif i==1:
                self.deconv.append(make_deconv_layers([feat_dim, feat_dim//2], norm_type='BN'))
            elif i==2:
                self.deconv.append(make_deconv_layers([feat_dim, feat_dim//2, feat_dim//4], norm_type='BN'))
            elif i==3:
                self.deconv.append(make_deconv_layers([feat_dim, feat_dim//2, feat_dim//4, feat_dim//8], norm_type='BN'))

    def forward(self, img_feat, lhand_bbox, rhand_bbox):
        '''
        : img_feat: [bs, c, cfg.output_hm_shape[2], cfg.output_hm_shape[1]]
        : lhand_bbox: (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        : rhand_bbox:
        '''
        lhand_bbox = torch.cat((torch.arange(lhand_bbox.shape[0]).float().cuda()[:, None], lhand_bbox),
                               1)  # batch_idx, xmin, ymin, xmax, ymax
        rhand_bbox = torch.cat((torch.arange(rhand_bbox.shape[0]).float().cuda()[:, None], rhand_bbox),
                               1)  # batch_idx, xmin, ymin, xmax, ymax
        hand_img_feats = []
        for i, deconv in enumerate(self.deconv):
            scale = 2**i
            img_feat_i = deconv(img_feat)
            lhand_bbox_roi = lhand_bbox.clone()
            lhand_bbox_roi[:, 1] = lhand_bbox_roi[:, 1] / cfg.input_body_shape[1] * cfg.output_hm_shape[2] * scale
            lhand_bbox_roi[:, 2] = lhand_bbox_roi[:, 2] / cfg.input_body_shape[0] * cfg.output_hm_shape[1] * scale
            lhand_bbox_roi[:, 3] = lhand_bbox_roi[:, 3] / cfg.input_body_shape[1] * cfg.output_hm_shape[2] * scale
            lhand_bbox_roi[:, 4] = lhand_bbox_roi[:, 4] / cfg.input_body_shape[0] * cfg.output_hm_shape[1] * scale
            assert (cfg.output_hm_shape[1]*scale, cfg.output_hm_shape[2]*scale) == (img_feat_i.shape[2], img_feat_i.shape[3])
            lhand_img_feat = roi_align(img_feat_i, lhand_bbox_roi,
                                                       (cfg.output_hand_hm_shape[1]*scale//2,
                                                        cfg.output_hand_hm_shape[2]*scale//2),
                                                       1.0, 0, 'avg', False)
            lhand_img_feat = torch.flip(lhand_img_feat, [3])  # flip to the right hand

            rhand_bbox_roi = rhand_bbox.clone()
            rhand_bbox_roi[:, 1] = rhand_bbox_roi[:, 1] / cfg.input_body_shape[1] * cfg.output_hm_shape[2] * scale
            rhand_bbox_roi[:, 2] = rhand_bbox_roi[:, 2] / cfg.input_body_shape[0] * cfg.output_hm_shape[1] * scale
            rhand_bbox_roi[:, 3] = rhand_bbox_roi[:, 3] / cfg.input_body_shape[1] * cfg.output_hm_shape[2] * scale
            rhand_bbox_roi[:, 4] = rhand_bbox_roi[:, 4] / cfg.input_body_shape[0] * cfg.output_hm_shape[1] * scale
            rhand_img_feat = roi_align(img_feat_i, rhand_bbox_roi,
                                                       (cfg.output_hand_hm_shape[1]*scale//2,
                                                        cfg.output_hand_hm_shape[2]*scale//2),
                                                       1.0, 0, 'avg', False)
            hand_img_feat = torch.cat((lhand_img_feat, rhand_img_feat))  # [bs, c, cfg.output_hand_hm_shape[2]*scale, cfg.output_hand_hm_shape[1]*scale]
            hand_img_feats.append(hand_img_feat)
        # print(len(hand_img_feats))
        return hand_img_feats[::-1]   # high resolution -> low resolution

class ImageHandRoI(nn.Module):
    def __init__(self):
        super(ImageHandRoI, self).__init__()

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
        return hand_img

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

class ScaleNet(nn.Module):
    def __init__(self):
        super(ScaleNet, self).__init__()
        self.scale_3d = nn.Parameter(torch.ones(1)*93.77, requires_grad=True)   # scale up the joint_3d to match the world coordinat

    def forward(self, img=None):
        return 0

def projectPoints(points, rmat, tvec, cameraMatrix, distCoeffs):

    # Apply rotation and translation to points
    points = torch.matmul(rmat, points.transpose(0, 1)) + tvec.unsqueeze(1)

    # Apply camera matrix and distortion coefficients
    x = points[0] / points[2]
    y = points[1] / points[2]
    r2 = x ** 2 + y ** 2
    k = 1 + distCoeffs[0] * r2 + distCoeffs[1] * r2 ** 2 + distCoeffs[4] * r2 ** 3
    x = x * k + 2 * distCoeffs[2] * x * y + distCoeffs[3] * (r2 + 2 * x ** 2)
    y = y * k + 2 * distCoeffs[3] * x * y + distCoeffs[2] * (r2 + 2 * y ** 2)
    u = cameraMatrix[0, 0] * x + cameraMatrix[0, 2]
    v = cameraMatrix[1, 1] * y + cameraMatrix[1, 2]

    # Return projected points
    return torch.stack([u, v], dim=1)

def batchProjectPoints(points, rmat, tvec, cameraMatrix, distCoeffs):

    # Apply rotation and translation to points
    points = torch.bmm(rmat, points.transpose(1, 2)) + tvec.unsqueeze(2)    # [4, 3, 67]

    # Apply camera matrix and distortion coefficients
    x = points[:, 0] / points[:, 2]
    y = points[:, 1] / points[:, 2]
    r2 = x ** 2 + y ** 2      # [4, 67]
    k = 1 + distCoeffs[:, 0:1] * r2 + distCoeffs[:, 1:2] * r2 ** 2 + distCoeffs[:, 4:5] * r2 ** 3
    x = x * k + 2 * distCoeffs[:,2:3] * x * y + distCoeffs[:, 3:4] * (r2 + 2 * x ** 2)
    y = y * k + 2 * distCoeffs[:,3:4] * x * y + distCoeffs[:, 2:3] * (r2 + 2 * y ** 2)

    u = cameraMatrix[:, 0:1, 0] * x + cameraMatrix[:, 0:1, 2]
    v = cameraMatrix[:, 1:2, 1] * y + cameraMatrix[:, 1:2, 2]
    return torch.stack([u, v], dim=2)

def draw2DKpt(img_paths, kpt_2ds):
    kpt_2ds = kpt_2ds.detach().cpu().numpy()
    for i in range(kpt_2ds.shape[0]):
        img_path = img_paths[i]
        img = cv2.imread(img_path)
        kpt_2d = kpt_2ds[i]
        for j in range(kpt_2d.shape[0]):
            cv2.circle(img, (int(kpt_2d[j, 0]), int(kpt_2d[j, 1])), 7, color=(255, 0, 0), thickness=-1)
        save_path = img_path.replace('/images/', '/images_project_kpt_vis/')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)
