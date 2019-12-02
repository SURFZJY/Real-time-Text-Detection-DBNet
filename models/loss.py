# -*- coding: utf-8 -*-

import itertools
import torch
from torch import nn
import numpy as np
import cv2

# import torchsnooper  ## for debug

class DBLoss(nn.Module):
    def __init__(self, alpha=1., beta=10., ohem_ratio=3):
        """
        Implement DB Loss.
        :param alpha: loss binary_map 前面的系数
        :param beta: loss threshold 前面的系数
        :param ohem_ratio: OHEM的比例
        """        
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ohem_ratio = ohem_ratio
          
    def forward(self, outputs, labels, training_masks, G_d):
        """
        Implement DB Loss.
        :param outputs: N 2 H W
        :param labels: N 2 H W
        :param training_masks: 
        """     
        prob_map = outputs[:, 0, :, :]
        thres_map = outputs[:, 1, :, :]
        gt_prob = labels[:, 0, :, :]
        gt_thres = labels[:, 1, :, :]
        
        G_d = G_d.to(dtype = torch.float32)
        training_masks = training_masks.to(dtype = torch.float32)
        
        # OHEM mask (todo)
        # selected_masks = self.ohem_batch(prob_map, gt_prob)
        # selected_masks = selected_masks.to(outputs.device)
        
        # 计算 prob loss
        loss_prob = self.dice_loss(prob_map, gt_prob, training_masks)
        # loss_prob = self.bce_loss(prob_map, gt_prob, selected_masks)
        
        # 计算 binary map loss
        bin_map = self.DB(prob_map, thres_map)
        loss_bin = self.dice_loss(bin_map, gt_prob, training_masks)
        # loss_prob = self.bce_loss(bin_map, gt_prob, selected_masks)
        
        # 计算 threshold map loss
        loss_fn = torch.nn.L1Loss(reduction='mean')
        L1_loss = loss_fn(thres_map, gt_thres)
        loss_thres = L1_loss * G_d 

        loss_prob = loss_prob.mean()
        loss_bin = loss_bin.mean()
        loss_thres = loss_thres.mean()

        loss_all = loss_prob + self.alpha * loss_bin + self.beta * loss_thres 
        return loss_all, loss_prob, loss_bin, loss_thres

    def DB(self, prob_map, thres_map, k=50):
        '''
        Differentiable binarization
        another form: torch.sigmoid(k * (prob_map - thres_map))
        '''
        return 1. / (torch.exp((-k * (prob_map - thres_map))) + 1)

    def dice_loss(self, pred_cls, gt_cls, training_mask):
        '''
        dice loss
        此处默认真实值和预测值的格式均为 NCHW
        :param gt_cls: 
        :param pred_cls: 
        :param training_mask: 
        :return:
        '''
        eps = 1e-5
        intersection = torch.sum(gt_cls * pred_cls * training_mask)
        union = torch.sum(gt_cls * training_mask) + torch.sum(pred_cls * training_mask) + eps
        loss = 1. - (2 * intersection / union)

        return loss

    def bce_loss(self, input, target, mask):
        if mask.sum() == 0:
            return torch.tensor(0.0, device=input.device, requires_grad=True)
        target[target <= 0.5] = 0
        target[target > 0.5] = 1
        input = input[mask.bool()]
        target = target[mask.bool()]
        loss = nn.BCELoss(reduction='mean')(input, target)
        return loss

    def ohem_single(self, score, gt_text):
        pos_num = (int)(np.sum(gt_text > 0.5))

        if pos_num == 0:
            selected_mask = np.zeros_like(score)
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_num = (int)(np.sum(gt_text <= 0.5))
        neg_num = (int)(min(pos_num * self.ohem_ratio, neg_num))

        if neg_num == 0:
            selected_mask = np.zeros_like(score)
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_score = score[gt_text <= 0.5]
        neg_score_sorted = np.sort(-neg_score)
        threshold = -neg_score_sorted[neg_num - 1]
        selected_mask = (score >= threshold) | (gt_text > 0.5)
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    def ohem_batch(self, scores, gt_texts):
        scores = scores.data.cpu().numpy()
        gt_texts = gt_texts.data.cpu().numpy()
        selected_masks = []
        for i in range(scores.shape[0]):
            selected_masks.append(self.ohem_single(scores[i, :, :], gt_texts[i, :, :]))

        selected_masks = np.concatenate(selected_masks, 0)
        selected_masks = torch.from_numpy(selected_masks).float()

        return selected_masks   
