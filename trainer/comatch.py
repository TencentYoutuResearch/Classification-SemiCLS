""" The Code is under Tencent Youtu Public Rule
"""
import argparse
import logging
import math
import os
from re import escape

import torch
import torch.nn.functional as F
from loss import builder as loss_builder

from .base_trainer import Trainer


class CoMatch(Trainer):
    """ Comatch trainer from https://arxiv.org/abs/2011.11183
        code migrated from https://github.com/salesforce/CoMatch
    """
    def __init__(self, cfg, device, all_cfg, **kwargs):
        super().__init__(cfg=cfg)
        # prepare self params
        self.all_cfg = all_cfg
        self.device = device
        self.amp = False
        if self.cfg.amp:
            from apex import amp
            self.amp = amp

        self.loss_x = loss_builder.build(cfg.loss_x)
        if self.amp:
            raise ValueError("Amp is not supported for comatch yet")
        self.pseudo_with_ema = False
        if self.all_cfg.get("ema", False):
            self.pseudo_with_ema = self.all_cfg.ema.get("pseudo_with_ema", False)

        # init params
        self.temperature = self.cfg.T
        self.alpha = self.cfg.alpha
        self.threshold = self.cfg.threshold
        self.num_classes = self.all_cfg.data.num_classes
        self.contrast_threshold = self.cfg.contrast_threshold
        self.lambda_u = self.cfg.lambda_u
        self.lambda_c = self.cfg.lambda_c

        # da setup use prob_list because the same name in official code
        self.prob_list = []
        self.da_len = self.cfg.get("da_len", 32)

        # initialize memory smoothed data
        self.queue_batch = self.cfg.get("queue_batch", 5)
        self.queue_size = self.queue_batch * (self.all_cfg.data.mu +
                                              1) * self.all_cfg.data.batch_size
        self.queue_feats = torch.zeros(self.queue_size,
                                       self.all_cfg.model.low_dim).to(device)
        self.queue_probs = torch.zeros(
            self.queue_size, self.num_classes).to(device)
        self.queue_ptr = 0

    def compute_loss(self,
                     data_x,
                     data_u,
                     model,
                     optimizer,
                     epoch,
                     iter,
                     ema_model=None,
                     task_specific_info=None,
                     **kwargs):
        try:
            self.queue_feats = task_specific_info['queue_feats']
            self.queue_probs = task_specific_info['queue_probs']
            self.queue_ptr = task_specific_info['queue_ptr']
        except KeyError:
            pass

        # make inputs
        inputs_x, targets_x = data_x
        inputs_x = inputs_x[0]
        inputs_u, targets_u = data_u
        inputs_u_w, inputs_u_s0, inputs_u_s1 = inputs_u

        # uniformat
        targets_x = targets_x.to(self.device)
        targets_u = targets_u.to(self.device)

        # prepare inputs
        batch_size = inputs_x.shape[0]
        batch_size_u = inputs_u_w.shape[0]
        inputs = torch.cat([inputs_x, inputs_u_w, inputs_u_s0, inputs_u_s1],
                           dim=0).to(self.device)

        # inference logits and features for projection head
        logits, features = model(inputs)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s0, logits_u_s1 = torch.split(logits[batch_size:], batch_size_u)

        feats_x = features[:batch_size]
        feats_u_w, feats_u_s0, feats_u_s1 = torch.split(features[batch_size:], batch_size_u)

        # supervision loss
        loss_x = self.loss_x(logits_x, targets_x)

        # other losses
        with torch.no_grad():
            logits_u_w = logits_u_w.detach()
            feats_x = feats_x.detach()
            feats_u_w = feats_u_w.detach()

            # pseudo label with weak aug
            probs = torch.softmax(logits_u_w, dim=1)

            # Distribution Alignment mentioned in paper
            self.prob_list.append(probs.mean(0))
            if len(self.prob_list) > self.da_len:
                self.prob_list.pop(0)
            prob_avg = torch.stack(self.prob_list, dim=0).mean(0)
            probs = probs / prob_avg
            probs = probs / probs.sum(dim=1, keepdim=True)

            probs_orig = probs.clone()

            # memory-smoothing using feature similairty
            if epoch > 0 or iter > self.queue_batch:
                A = torch.exp(
                    torch.mm(feats_u_w, self.queue_feats.t()) /
                    self.temperature)
                A = A / A.sum(1, keepdim=True)
                probs = self.alpha * probs + (1 - self.alpha) * torch.mm(
                    A, self.queue_probs)

            # get pseudo label and mask
            # note here the label is soft, hard label is for acc calculation
            scores, lbs_u_guess = torch.max(probs, dim=1)
            mask = scores.ge(self.threshold).float()

            # update memory bank
            feats_w = torch.cat([feats_u_w, feats_x], dim=0)
            onehot = torch.zeros(batch_size,
                                 self.num_classes).to(self.device).scatter(
                                     1, targets_x.view(-1, 1), 1)
            probs_w = torch.cat([probs_orig, onehot], dim=0)

            n = batch_size + batch_size_u
            self.queue_feats[self.queue_ptr:self.queue_ptr + n, :] = feats_w
            self.queue_probs[self.queue_ptr:self.queue_ptr + n, :] = probs_w
            self.queue_ptr = (self.queue_ptr + n) % self.queue_size

        # embedding similarity
        sim = torch.exp(torch.mm(feats_u_s0, feats_u_s1.t())/self.temperature)
        sim_probs = sim / sim.sum(1, keepdim=True)

        # pseudo-label graph with self-loop for contrustive similarity
        Q = torch.mm(probs, probs.t())
        Q.fill_diagonal_(1)
        pos_mask = (Q >= self.contrast_threshold).float()

        Q = Q * pos_mask
        Q = Q / Q.sum(1, keepdim=True)
        # contrastive loss
        loss_contrast = - (torch.log(sim_probs + 1e-7) * Q).sum(1)
        loss_contrast = loss_contrast.mean()

        # unsupervised classification loss, cross entropy but author self implemented
        loss_u = -torch.sum(
            (F.log_softmax(logits_u_s0, dim=1) * probs), dim=1) * mask
        loss_u = loss_u.mean()

        loss = loss_x + self.lambda_u * loss_u + self.lambda_c * loss_contrast

        loss.backward()

        # calculate pseudo label acc
        right_labels = (lbs_u_guess == targets_u).float() * mask
        pseudo_label_acc = right_labels.sum() / max(mask.sum(), 1.0)

        # modify task_specific_info in place
        task_specific_info['queue_feats'] = self.queue_feats
        task_specific_info['queue_probs'] = self.queue_probs
        task_specific_info['queue_ptr'] = self.queue_ptr

        # output loss
        loss_dict = {
            "loss": loss,
            "loss_x": loss_x,
            "loss_u": loss_u,
            "loss_c": loss_contrast,
            "mask_prob": mask.mean(),
            "pseudo_acc": pseudo_label_acc,
        }
        return loss_dict
