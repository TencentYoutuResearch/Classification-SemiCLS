""" The Code is under Tencent Youtu Public Rule
ckpt related utils
"""
import os
import shutil

import torch

"""
    save_ckpt_dict = {
        'epoch': epoch + 1,
        'state_dict': model_to_save.state_dict(),
        'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
        'acc': test_acc,
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
"""

    # save_ckpt_dict = {
    #     'epoch': epoch + 1,
    #     'state_dict': model_to_save.state_dict(),
    #     'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
    #     'acc': test_acc,
    #     'best_acc': best_acc,
    #     'optimizer': optimizer.state_dict(),
    #     'scheduler': scheduler.state_dict(),
    # }


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def save_ckpt_dict(args, model, ema_model, epoch, test_acc, optimizer,
                   scheduler, task_specific_info, is_best, best_acc):
    model_to_save = model.module if hasattr(model, "module") else model
    if args.use_ema:
        ema_to_save = ema_model.ema.module if hasattr(
            ema_model.ema, "module") else ema_model.ema
    save_ckpt_dict = {
        'epoch': epoch + 1,
        'state_dict': model_to_save.state_dict(),
        'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
        'acc': test_acc,
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    save_ckpt_dict.update(task_specific_info)
    save_checkpoint(save_ckpt_dict, is_best, args.out)
