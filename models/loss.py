# -*- coding: utf-8 -*-

import torch
from torch import nn


class DBLoss(nn.Module):
    def __init__(self, alpha=1., beta=10., ohem_ratio=3, reduction='mean'):
        """
        Implement DB Loss.
        :param alpha: binary map loss 前面的系数
        :param beta: threshold map loss 前面的系数
        :param ohem_ratio: OHEM的比例 (正负样本比 1:ohem_ratio)
        :param reduction: loss reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction

    def forward(self, outputs, labels, training_masks, G_d):
        """
        Implement DB Loss.
        :param outputs: N 2 H W (prob_map, thres_map)
        :param labels: N 2 H W (gt_prob, gt_thres)
        :param training_masks: N H W
        :param G_d: N H W (dilated text region mask for threshold loss)
        """
        prob_map = outputs[:, 0, :, :]
        thres_map = outputs[:, 1, :, :]
        gt_prob = labels[:, 0, :, :]
        gt_thres = labels[:, 1, :, :]

        G_d = G_d.to(dtype=torch.float32)
        training_masks = training_masks.to(dtype=torch.float32)

        # OHEM mask for probability and binary map loss (GPU-friendly)
        selected_masks = self.ohem_batch(prob_map, gt_prob, training_masks)
        selected_masks = selected_masks.to(outputs.device)

        # 计算 prob loss (BCE + OHEM, 与官方实现一致)
        loss_prob = self.bce_loss(prob_map, gt_prob, selected_masks)

        # 计算 binary map loss (differentiable binarization + BCE + OHEM)
        bin_map = self.DB(prob_map, thres_map)
        loss_bin = self.bce_loss(bin_map, gt_prob, selected_masks)

        # 计算 threshold map loss (L1, only within dilated text regions)
        # 修复: 逐像素L1 * mask, 再求均值 (之前错误地先求mean再乘mask)
        eps = 1e-6
        loss_thres = (torch.abs(thres_map - gt_thres) * G_d).sum() / (G_d.sum() + eps)

        loss_prob = loss_prob.mean()
        loss_bin = loss_bin.mean()

        loss_all = loss_prob + self.alpha * loss_bin + self.beta * loss_thres
        return loss_all, loss_prob, loss_bin, loss_thres

    def DB(self, prob_map, thres_map, k=50):
        """Differentiable binarization"""
        return torch.reciprocal(1. + torch.exp(-k * (prob_map - thres_map)))

    def dice_loss(self, pred_cls, gt_cls, training_mask):
        """dice loss (kept as alternative)"""
        eps = 1e-5
        intersection = torch.sum(gt_cls * pred_cls * training_mask)
        union = torch.sum(gt_cls * training_mask) + torch.sum(pred_cls * training_mask) + eps
        loss = 1. - (2 * intersection / union)
        return loss

    def bce_loss(self, input, target, mask):
        """BCE loss with mask support"""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=input.device, requires_grad=True)
        target = target.clone()
        target[target <= 0.5] = 0
        target[target > 0.5] = 1
        input = input[mask.bool()]
        target = target[mask.bool()]
        loss = nn.BCELoss(reduction='mean')(input, target)
        return loss

    def ohem_single(self, score, gt_text, training_mask):
        """
        Online Hard Example Mining for a single sample.
        Fully implemented in PyTorch (GPU-friendly, no numpy conversion).
        """
        pos_mask = (gt_text > 0.5) & (training_mask > 0.5)
        neg_mask = (gt_text <= 0.5) & (training_mask > 0.5)

        pos_num = pos_mask.sum().item()
        neg_num = neg_mask.sum().item()

        if pos_num == 0:
            return training_mask.bool()

        neg_num = min(int(pos_num * self.ohem_ratio), neg_num)
        if neg_num == 0:
            return pos_mask

        neg_score = score[neg_mask]
        # select hardest negatives (highest predicted scores)
        neg_score_sorted, _ = torch.sort(neg_score, descending=True)
        threshold = neg_score_sorted[neg_num - 1]
        selected_mask = ((score >= threshold) & neg_mask) | pos_mask
        return selected_mask

    def ohem_batch(self, scores, gt_texts, training_masks):
        """
        Online Hard Example Mining for a batch.
        Runs on GPU without numpy conversion.
        """
        selected_masks = []
        for i in range(scores.shape[0]):
            selected_mask = self.ohem_single(
                scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]
            )
            selected_masks.append(selected_mask.unsqueeze(0))

        selected_masks = torch.cat(selected_masks, 0).float()
        return selected_masks
