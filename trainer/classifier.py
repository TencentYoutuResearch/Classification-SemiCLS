""" The Code is under Tencent Youtu Public Rule
"""
from loss import builder as loss_builder

from .base_trainer import Trainer
# trainer "Classifier"
# used for supervised learning
# and it only takes inputs_x

"""
trainer "Classifier"
used for supervised learning
and it only takes inputs_x

"""
class Classifier(Trainer):
    def __init__(self, cfg, device, all_cfg, **kwargs):
        super().__init__(cfg=cfg)

        self.all_cfg = all_cfg
        self.device = device
        self.loss_x = loss_builder.build(cfg.loss_x)

    def compute_loss(self, data_x, model, **kwargs):
        # make inputs
        inputs_x, targets_x = data_x
        targets_x = targets_x.to(self.device)
        #inputs_x is list type because of list transform
        inputs_x = inputs_x[0]
        logits = model(inputs_x.to(self.device))
        loss = self.loss_x(logits, targets_x)

        loss.backward()

        # calculate pseudo label acc
        loss_dict = {
            "loss": loss,
        }
        return loss_dict
