import math
import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class RLELoss_poseur_old(nn.Module):
    ''' RLE Regression Loss
    '''

    def __init__(self, OUTPUT_3D=False, use_target_weight=True, size_average=True):
        super(RLELoss_poseur_old, self).__init__()
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, output, target_uv, target_uv_weight):

        pred_jts = output.pred_jts
        sigma = output.sigma
        gt_uv = target_uv.reshape(pred_jts.shape)
        gt_uv_weight = target_uv_weight.reshape(pred_jts.shape)



        nf_loss = output.nf_loss * gt_uv_weight[:, :, :1]
        # print(gt_uv.min(), gt_uv.max())

        residual = True
        if residual:
            Q_logprob = self.logQ(gt_uv, pred_jts, sigma) * gt_uv_weight
            loss = nf_loss + Q_logprob

        if self.size_average and gt_uv_weight.sum() > 0:
            return loss.sum() / len(loss)
        else:
            return loss.sum()

@LOSSES.register_module()
class RLELoss_poseur(nn.Module):
    ''' RLE Regression Loss
    '''

    def __init__(self, OUTPUT_3D=False, use_target_weight=True, size_average=True):
        super(RLELoss_poseur, self).__init__()
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, output, target_uvd, target_uvd_weight):

        pred_jts = output.pred_jts
        sigma = output.sigma
        gt_uv = target_uvd.reshape(pred_jts.shape)
        gt_uv_weight = target_uvd_weight.reshape(pred_jts.shape)

        # nf_loss = output.nf_loss * gt_uv_weight[:, :, :1]
        nf_loss = output.nf_loss * gt_uv_weight

        residual = True
        if residual:
            Q_logprob = self.logQ(gt_uv, pred_jts, sigma) * gt_uv_weight
            loss = nf_loss + Q_logprob

        if self.size_average and gt_uv_weight.sum() > 0:
            return loss.sum() / len(loss)
        else:
            return loss.sum()

@LOSSES.register_module()
class RLEOHKMLoss(nn.Module):
    ''' RLE Regression Loss
    '''

    def __init__(self, OUTPUT_3D=False, use_target_weight=True, size_average=True, topk=8, 
                    ori_weight = 1.0, ohkm_weight = 0.0):
        super(RLEOHKMLoss, self).__init__()
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)
        self.topk = topk
        self.ori_weight = ori_weight
        self.ohkm_weight = ohkm_weight
        self.neg_inf = -float("Inf")

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def ohkm(self, loss, weight):
        # mask = weight == 0
        loss_value = loss.clone().detach()
        loss_value[weight == 0] = self.neg_inf
        _, topk_idx = torch.topk(
            loss_value, k=self.topk, dim=1, sorted=False)
        tmp_loss = torch.gather(loss, 1, topk_idx)
        tmp_weight = torch.gather(weight, 1, topk_idx)
        # tmp_loss[tmp_loss==-float("Inf")] = 0
        tmp_loss = tmp_loss * tmp_weight
        tmp_loss = tmp_loss.flatten(start_dim=1).sum(dim = 1)
        # tmp_weight = tmp_weight.flatten(start_dim=1).sum(dim = 1)
        # tmp_loss = tmp_loss / tmp_weight

        return tmp_loss.mean()

    def ori(self, loss, weight):
        # mask = weight == 0
        loss = loss * weight
        loss = loss.flatten(start_dim=1).sum(dim = 1)
        # weight = weight.flatten(start_dim=1).sum(dim = 1)

        return loss.mean()

    def forward(self, output, target_uv, target_uv_weight):

        pred_jts = output.pred_jts
        sigma = output.sigma
        gt_uv = target_uv.reshape(pred_jts.shape)
        gt_uv_weight = target_uv_weight.reshape(pred_jts.shape)

        # gt_uv_weight = gt_uv_weight[:, :, :1] 
        nf_loss = output.nf_loss
        q_loss = self.logQ(gt_uv, pred_jts, sigma)

        # nf_loss_ohkm = self.ohkm(nf_loss, gt_uv_weight)
        # q_loss_ohkm = self.ohkm(q_loss, gt_uv_weight)

        ori_loss = nf_loss + q_loss
        ohkm_loss = self.ohkm(ori_loss, gt_uv_weight)
        ori_loss = self.ori(ori_loss, gt_uv_weight)

        loss = self.ori_weight * ori_loss + self.ohkm_weight * ohkm_loss
        return loss #TODO mean?


        # nf_loss = output.nf_loss * gt_uv_weight


        # Q_logprob = self.logQ(gt_uv, pred_jts, sigma) * gt_uv_weight
        # loss = nf_loss + Q_logprob

        # return loss.sum() / len(loss)


@LOSSES.register_module()
class RLELoss3D(nn.Module):
    ''' RLE Regression Loss 3D
    '''

    def __init__(self, OUTPUT_3D=False, size_average=True):
        super(RLELoss3D, self).__init__()
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, output, labels):
        nf_loss = output.nf_loss
        pred_jts = output.pred_jts
        sigma = output.sigma
        gt_uv = labels['target_uvd'].reshape(pred_jts.shape)
        gt_uv_weight = labels['target_uvd_weight'].reshape(pred_jts.shape)
        nf_loss = nf_loss * gt_uv_weight

        residual = True
        if residual:
            Q_logprob = self.logQ(gt_uv, pred_jts, sigma) * gt_uv_weight
            loss = nf_loss + Q_logprob

        if self.size_average and gt_uv_weight.sum() > 0:
            return loss.sum() / len(loss)
        else:
            return loss.sum()