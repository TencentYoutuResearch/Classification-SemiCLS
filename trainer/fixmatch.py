""" The Code is under Tencent Youtu Public Rule
Self-implemented FixMatch
The performance is similar to FixMatch. Thanks to https://github.com/kekmodel/FixMatch-pytorch.
The implementation is based on FixMatch-pytorch.
"""

import argparse
import logging
import math
import os

import torch
import torch.nn.functional as F
from loss import builder as loss_builder

from .base_trainer import Trainer


class FixMatch(Trainer):

    def __init__(self, cfg, device, all_cfg, **kwargs):
        super().__init__(cfg=cfg)

        self.all_cfg = all_cfg
        self.device = device
        if self.cfg.amp:
            from apex import amp
            self.amp = amp
        self.loss_x = loss_builder.build(cfg.loss_x)
        self.loss_u = loss_builder.build(cfg.loss_u)

        # pseudo with ema, this will intrige bad results and not used in paper
        self.pseudo_with_ema = False
        self.da = False
        self._get_config()

    def _get_config(self):
        if self.all_cfg.get("ema", False):
            self.pseudo_with_ema = self.all_cfg.ema.get(
                "pseudo_with_ema", False)

        # distribution alignment mentioned in paper
        self.prob_list = []
        if self.cfg.get("DA", False):
            self.da = self.cfg.DA.use

    def _da_pseudo_label(self, prob_list, logits_u_w):
        """ distribution alignment
        """
        with torch.no_grad():
            probs = torch.softmax(logits_u_w, dim=1)

            prob_list.append(probs.mean(0))
            if len(prob_list) > self.cfg.DA.da_len:
                prob_list.pop(0)
            prob_avg = torch.stack(prob_list, dim=0).mean(0)
            probs = probs / prob_avg
            probs = probs / probs.sum(dim=1, keepdim=True)
            probs = probs.detach()
        return probs

    def _get_pseudo_label_acc(self, p_targets_u, mask, targets_u):
        targets_u = targets_u.to(self.device)
        right_labels = (p_targets_u == targets_u).float() * mask
        pseudo_label_acc = right_labels.sum() / max(mask.sum(), 1.0)
        return pseudo_label_acc

    def _get_psuedo_label_and_mask(self, probs_u_w):
        max_probs, p_targets_u = torch.max(probs_u_w, dim=-1)
        mask = max_probs.ge(self.cfg.threshold).float()
        return p_targets_u, mask

    def compute_loss(self,
                     data_x,
                     data_u,
                     model,
                     optimizer,
                     ema_model=None,
                     **kwargs):
        # make inputs
        inputs_x, targets_x = data_x
        inputs_x = inputs_x[0]
        inputs_u, targets_u = data_u
        inputs_u_w, inputs_u_s = inputs_u

        batch_size = inputs_x.shape[0]
        targets_x = targets_x.to(self.device)

        # whether use ema on pseudo labels, the performance
        # is lower according to our experiments
        if self.pseudo_with_ema:
            # for ema pseudo label
            logits_x = model(inputs_x.to(self.device))
            logits_u_s = model(inputs_u_s.to(self.device))

            logits_u_w = ema_model(inputs_u_w.to(self.device))
        else:
            inputs = torch.cat([inputs_x, inputs_u_w, inputs_u_s],
                               dim=0).to(self.device)
            logits = model(inputs)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

        # supervised loss
        Lx = self.loss_x(logits_x, targets_x, reduction='mean')

        # whether use da for pseudo labels, the performance is also lower based on our
        # performance
        if not self.da:
            probs_u_w = torch.softmax(logits_u_w.detach() / self.cfg.T, dim=-1)
        else:
            probs_u_w = self._da_pseudo_label(self.prob_list, logits_u_w)

        # making pseudo labels
        p_targets_u, mask = self._get_psuedo_label_and_mask(probs_u_w)

        # semi-supervised loss
        Lu = (self.loss_u(logits_u_s, p_targets_u, reduction='none') *
              mask).mean()

        loss = Lx + self.cfg.lambda_u * Lu

        # whether to use amp for accelerated training
        if hasattr(self, "amp"):
            with self.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        elif "SCALER" in kwargs and kwargs["SCALER"] is not None:
            kwargs['SCALER'].scale(loss).backward()
        else:
            loss.backward()

        # calculate pseudo label acc
        # targets_u = targets_u.to(self.device)
        # right_labels = (p_targets_u == targets_u).float() * mask
        # pseudo_label_acc = right_labels.sum() / max(mask.sum(), 1.0)
        pseudo_label_acc = self._get_pseudo_label_acc(p_targets_u, mask, targets_u)

        loss_dict = {
            "loss": loss,
            "loss_x": Lx,
            "loss_u": Lu,
            "mask_prob": mask.mean(),
            "pseudo_acc": pseudo_label_acc,
        }
        return loss_dict
