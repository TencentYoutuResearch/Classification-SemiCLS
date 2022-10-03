""" The Code is under Tencent Youtu Public Rule
SSL trainer FixMatch+CCSSL

cfg.contrast_with_labeled(bool): use labeled data on contrastive branch
cfg.contrast_with_threshold(bool): use threshold to filter out low confidence samples
cfg.contrast_with_softlabel(bool): use confidence score for re-weighting
"""
import argparse
import copy
import logging
import math
import os
import pdb

import torch
import torch.nn.functional as F
from loss import builder as loss_builder
from loss.soft_supconloss import SoftSupConLoss

from .base_trainer import Trainer


class FixMatchCCSSL(Trainer):
    """ FixMatch CCSSL class based on FixMatch
    """
    def __init__(self, cfg, device, all_cfg, **kwargs):
        super().__init__(cfg=cfg)

        self.all_cfg = all_cfg
        self.device = device
        if self.cfg.amp:
            from apex import amp
            self.amp = amp

        self.contrast_with_labeled = self.all_cfg.get("contrast_with_labeled",
                                                      False)
        self.contrast_with_threshold = self.cfg.get("contrast_with_threshold",
                                                    False)
        self.contrast_with_softlabel = self.cfg.get("contrast_with_softlabel",
                                                    False)
        # loss for supervised branch
        self.loss_x = loss_builder.build(cfg.loss_x)
        # loss for unsupervised branch
        self.loss_u = loss_builder.build(cfg.loss_u)

        # to use background re-weighting, will releaase SupConLoss Later
        # if self.contrast_with_softlabel:
        # self.loss_contrast = SoftSupConLoss(
        #     temperature=self.cfg.temperature)
        # else:
        #     self.loss_contrast = SupConLoss(temperature=self.cfg.temperature)
        self.loss_contrast = SoftSupConLoss(temperature=self.cfg.temperature)

        # pseudo with ema, this will intrige bad results and not used in paper
        self.pseudo_with_ema = False
        if self.all_cfg.get("ema", False):
            self.pseudo_with_ema = self.all_cfg.ema.get(
                "pseudo_with_ema", False)

        # distribution alignment mentioned in paper
        self.da = False
        self.prob_list = []
        if self.cfg.get("DA", False):
            self.da = self.cfg.DA.use

    # Distribution Alignment for pseudo label
    def _da_pseudo_label(self, logits_u_w):
        with torch.no_grad():
            probs = torch.softmax(logits_u_w, dim=1)

            self.prob_list.append(probs.mean(0))
            if len(self.prob_list) > self.cfg.DA.da_len:
                self.prob_list.pop(0)
            prob_avg = torch.stack(self.prob_list, dim=0).mean(0)
            probs = probs / prob_avg
            probs = probs / probs.sum(dim=1, keepdim=True)
            probs = probs.detach()
        return probs

    def compute_loss(self,
                     data_x,
                     data_u,
                     model,
                     optimizer,
                     ema_model=None,
                     **kwargs):
        # make inputs
        #pdb.set_trace()
        inputs_x, targets_x = data_x
        inputs_x_w = inputs_x[0]

        inputs_u, targets_u = data_u
        inputs_u_w, inputs_u_s, inputs_u_s1 = inputs_u

        batch_size = inputs_x_w.shape[0]
        targets_x = targets_x.to(self.device)

        if self.pseudo_with_ema:
            # for ema pseudo label
            logits_x, _ = model(inputs_x_w.to(self.device))
            logits_u_s, _ = model(inputs_u_s.to(self.device))
            logits_u_w, _ = ema_model(inputs_u_w.to(self.device))
            raise ValueError("Unsupported psuedo with ema mode in exp type")
        else:
            # inference once for all
            inputs = torch.cat(
                [inputs_x_w, inputs_u_w, inputs_u_s, inputs_u_s1],
                dim=0).to(self.device)
            logits, features = model(inputs)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s, _ = logits[batch_size:].chunk(3)
            _, f_u_s1, f_u_s2 = features[batch_size:].chunk(3)
            del logits
            del features
            del _

        Lx = self.loss_x(logits_x, targets_x, reduction='mean')
        if not self.da:
            probs_u_w = torch.softmax(logits_u_w.detach() / self.cfg.T, dim=-1)
        else:
            probs_u_w = self._da_pseudo_label(logits_u_w)

        # pseudo label and scores for u_w
        max_probs, p_targets_u = torch.max(probs_u_w, dim=-1)
        # filter out low confidence pseudo label by self.cfg.threshold
        mask = max_probs.ge(self.cfg.threshold).float()
        Lu = (self.loss_u(logits_u_s, p_targets_u, reduction='none') *
              mask).mean()

        # for supervised contrastive
        labels = p_targets_u
        features = torch.cat([f_u_s1.unsqueeze(1), f_u_s2.unsqueeze(1)], dim=1)

        # In case of early training stage, pseudo labels have low scores
        if labels.shape[0] != 0:
            if self.contrast_with_softlabel:
                select_matrix = None
                if self.cfg.get("contrast_left_out", False):
                    with torch.no_grad():
                        select_matrix = self.contrast_left_out(max_probs)
                    Lcontrast = self.loss_contrast(features,
                                                   max_probs,
                                                   labels,
                                                   select_matrix=select_matrix)

                elif self.cfg.get("contrast_with_thresh", False):
                    contrast_mask = max_probs.ge(
                        self.cfg.contrast_with_thresh).float()
                    Lcontrast = self.loss_contrast(features,
                                                   max_probs,
                                                   labels,
                                                   reduction=None)
                    Lcontrast = (Lcontrast * contrast_mask).mean()

                else:
                    Lcontrast = self.loss_contrast(features, max_probs, labels)
            else:
                if self.cfg.get("contrast_left_out", False):
                    with torch.no_grad():
                        select_matrix = self.contrast_left_out(max_probs)
                    Lcontrast = self.loss_contrast(features,
                                                   labels,
                                                   select_matrix=select_matrix)
                else:
                    Lcontrast = self.loss_contrast(features, labels)

        else:
            Lcontrast = sum(features.view(-1, 1)) * 0

        loss = Lx + self.cfg.lambda_u * Lu + self.cfg.lambda_contrast * Lcontrast

        if hasattr(self, "amp"):
            with self.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        elif "SCALER" in kwargs and kwargs["SCALER"] is not None:
            kwargs['SCALER'].scale(loss).backward()
        else:
            loss.backward()

        # calculate pseudo label acc
        targets_u = targets_u.to(self.device)
        right_labels = (p_targets_u == targets_u).float() * mask
        pseudo_label_acc = right_labels.sum() / max(mask.sum(), 1.0)

        loss_dict = {
            "loss": loss,
            "loss_x": Lx,
            "loss_u": Lu,
            "loss_contrast": Lcontrast,
            "mask_prob": mask.mean(),
            "pseudo_acc": pseudo_label_acc,
        }

        # if self.cfg.get("contrast_left_out", False):
        #     loss_dict.update({"contrast_mask": contrast_mask.mean()})

        return loss_dict

    def contrast_left_out(self, max_probs):
        """contrast_left_out

        If contrast_left_out, will select positive pairs based on
            max_probs > contrast_with_thresh, others will set to 0
            later max_probs will be used to re-weight the contrastive loss

        Args:
            max_probs (torch Tensor): prediction probabilities

        Returns:
            select_matrix: select_matrix with probs < contrast_with_thresh set
                to 0
        """
        contrast_mask = max_probs.ge(self.cfg.contrast_with_thresh).float()
        contrast_mask2 = torch.clone(contrast_mask)
        contrast_mask2[contrast_mask == 0] = -1
        select_elements = torch.eq(contrast_mask2.reshape([-1, 1]),
                                   contrast_mask.reshape([-1, 1]).T).float()
        select_elements += torch.eye(contrast_mask.shape[0]).to(self.device)
        select_elements[select_elements > 1] = 1
        select_matrix = torch.ones(contrast_mask.shape[0]).to(
            self.device) * select_elements
        return select_matrix
