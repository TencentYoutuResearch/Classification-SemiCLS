""" The Code is under Tencent Youtu Public Rule

this is the mian sript in our SSL tool box
The interface contains many args

$CONFIG_PATH=/your/config/path
$OUTPUT_PATH=/your/output/path
For single gpu
number_of_gpus=1
python3 train_semi.py --cfg $CONFIG_PATH  --out $OUTPUT_PATH --seed 5

For multi gpu
number_of_gpus=N
python3 -m torch.distributed.launch --nproc_per_node $number_of_gpus \
    train_semi.py\
        --cfg $CONFIG_PATH
        --out $OUTPUT_PATH --use_BN True  --seed 5

"""
import argparse
import logging
import math
import os
import random
import time
from calendar import c
from sched import scheduler

import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import builder as dataset_builder
from models import builder as model_builder
from optimizer import builder as optim_builder
from scheduler import builder as scheduler_builder
from trainer import builder as trainer_builder
from utils import AverageMeter, AverageMeterManeger, accuracy
from utils.ckpt_utils import save_ckpt_dict
from utils.config_utils import overwrite_config
from utils.log_utils import get_default_logger

torch.autograd.set_detect_anomaly(True)

# global variables
global logger
best_acc = 0
SCALER = None


def set_seed(args):
    """ set seed for the whole program for removing randomness
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_test_model(ema_model, model, use_ema):
    """ use ema model or test model
    """
    if use_ema:
        test_model = ema_model.ema
        test_prefix = "ema"
    else:
        test_model = model
        test_prefix = ""
    return test_model, test_prefix


def main():
    args = get_args()
    global best_acc

    # prepare config and make output dir
    cfg = Config.fromfile(args.cfg)
    cfg = overwrite_config(cfg, args.other_args)
    cfg.resume = args.resume
    cfg.data['eval_step'] = cfg.train.eval_step

    # set amp scaler, usually no use
    global SCALER
    if args.fp16:
        SCALER = torch.cuda.amp.GradScaler()
    else:
        SCALER = None

    # set summary writer on rank 0
    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)
        cfg.dump(os.path.join(args.out, os.path.basename(args.cfg)))

    # set up logger
    global logger
    logger = get_default_logger(
        args=args,
        logger_name='root',
        default_level=logging.INFO
        if args.local_rank in [-1, 0] else logging.WARN,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        rank=args.local_rank)

    #set random seed
    if args.seed is not None:
        cfg.seed = args.seed
    elif cfg.get("seed", None) is not None:
        args.seed = cfg.seed

    # set folds for stl10 dataset if used
    if "folds" in cfg.data.keys():
        cfg.data.folds = cfg.seed

    args.amp = False
    if cfg.get("amp", False) and cfg.amp.use:
        args.amp = True
        args.opt_level = cfg.amp.opt_level

    if not args.pretrained and "pretrained" in cfg.keys():
        args.pretrained = cfg.pretrained

    args.total_steps = cfg.train.total_steps
    args.eval_steps = cfg.train.eval_step

    # init dist params
    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    # set device
    args.device = device

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}", )

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # make dataset
    #labeled_dataset, unlabeled_dataset, test_dataset = dataset_builder.build(cfg.data)
    labeled_dataset, unlabeled_dataset, test_dataset = dataset_builder.build(
        cfg.data)

    if args.local_rank == 0:
        torch.distributed.barrier()

    # make dataset loader
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader, unlabeled_trainloader, test_loader = get_dataloader(
        cfg, train_sampler, labeled_dataset, unlabeled_dataset, test_dataset)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = model_builder.build(cfg.model)

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q'
                                ) and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            print("Missing keys", msg.missing_keys)

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    # make optimizer,scheduler
    optimizer = optim_builder.build(cfg.optimizer, model)
    scheduler = set_scheduler(cfg, args, optimizer)

    # set ema
    args.use_ema = False
    ema_model = None
    if cfg.get("ema", False) and cfg.ema.use:
        args.use_ema = True
        from models.ema import ModelEMA
        ema_model = ModelEMA(args.device, model, cfg.ema.decay)

    args.start_epoch = 0

    # initialize from resume for fixed info and task_specific_info
    task_specific_info = dict()

    if args.resume:
        if args.use_ema:
            resume(args,
                   model,
                   optimizer,
                   scheduler,
                   task_specific_info,
                   ema_model=ema_model)
        else:
            resume(args, model, optimizer, scheduler, task_specific_info)

    # builde model trainer
    cfg.train.trainer['amp'] = args.amp
    model_trainer = trainer_builder.build(cfg.train.trainer)(device=device,
                                                             all_cfg=cfg)

    # process model
    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level=args.opt_level)
    #use_BN
    use_batchnorm = args.use_BN
    if use_batchnorm:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True)

    log_info(cfg, args)

    model.zero_grad()
    #train loop
    train(args, cfg, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, model_trainer,
          task_specific_info)


# resume from checkpoint
def resume(args,
           model,
           optimizer,
           scheduler,
           task_specific_info,
           ema_model=None):
    logger.info("==> Resuming from checkpoint..")
    if not os.path.isfile(args.resume):
        logger.info("Error resuming from {}! try resume from last one".format(
            args.resume))
        args.resume = os.path.join(args.out, "checkpoint.pth.tar")
        logger.info("Resuming from {}".format(args.resume))

    # try resume if specified
    if not os.path.isfile(args.resume):
        logger.info("Failed to resume from {}".format(args.resume))
    else:
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        for key in checkpoint.keys():
            if key not in [
                    'epoch', 'state_dict', 'ema_state_dict', 'acc', 'best_acc',
                    'optimizer', 'scheduler'
            ]:
                task_specific_info[key] = checkpoint[key]
                try:
                    task_specific_info[key] = task_specific_info[key].to(
                        args.device)
                except:
                    pass


#set scheduler
def set_scheduler(cfg, args, optimizer):
    args.epochs = math.ceil(cfg.train.total_steps / cfg.train.eval_step)
    args.eval_step = cfg.train.eval_step
    args.total_steps = cfg.train.total_steps
    scheduler = scheduler_builder.build(cfg.scheduler)(optimizer=optimizer)

    return scheduler


# log info before training
def log_info(cfg, args):
    logger.info("***** Running training *****")
    logger.info(f"  Task = {cfg.data.type}")
    if "num_labeled" in cfg.data.keys():
        logging_num_labeled = cfg.data.num_labeled
    elif "percent" in cfg.data.keys():
        logging_num_labeled = "{}%".format(cfg.data.percent)
    else:
        logging_num_labeled = "seed {}".format(cfg.seed)

    logger.info(f"  Num Label = {logging_num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {cfg.data.batch_size}")
    logger.info(
        f"  Total train batch size = {cfg.data.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")
    logger.info(cfg)


# get args
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--cfg', type=str, required=True, help='a config')
    parser.add_argument('--gpu-id',
                        default='0',
                        type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--out',
                        default='result',
                        help='directory to output the result')
    parser.add_argument('--pretrained',
                        default=None,
                        help='directory to pretrained model')
    parser.add_argument('--resume',
                        default='',
                        type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int, help="random seed")
    parser.add_argument('--use_BN',
                        default=False,
                        type=bool,
                        help="use_batchnorm")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="whether use fp16 for training")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress',
                        action='store_true',
                        help="don't use progress bar")
    parser.add_argument(
        '--other-args',
        default='',
        type=str,
        help='other args to overwrite the config, keys are split \
                            by space and args split by |, such as \'seed 1|train trainer T 1|\' '
    )

    args = parser.parse_args()
    return args


# labeled_trainloader,unlabeled_trainloader,test_loader
def get_dataloader(cfg, train_sampler, labeled_dataset, unlabeled_dataset,
                   test_dataset):
    # prepare labeled_trainloader
    labeled_trainloader = DataLoader(labeled_dataset,
                                     sampler=train_sampler(labeled_dataset),
                                     batch_size=cfg.data.batch_size,
                                     num_workers=cfg.data.num_workers,
                                     drop_last=True)
    # prepare unlabeled_trainloader
    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=cfg.data.batch_size * cfg.data.mu,
        num_workers=cfg.data.num_workers,
        drop_last=True)
    # prepare test_loader
    test_loader = DataLoader(test_dataset,
                             sampler=SequentialSampler(test_dataset),
                             batch_size=cfg.data.batch_size,
                             num_workers=cfg.data.num_workers)

    return labeled_trainloader, unlabeled_trainloader, test_loader


# train_loop
def train(args, cfg, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, model_trainer,
          task_specific_info):
    """
    Train function for training
    """
    global best_acc
    test_accs = []

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    for epoch in range(args.start_epoch, args.epochs):
        # init logger
        meter_manager = AverageMeterManeger()
        meter_manager.register('batch_time')
        meter_manager.register('data_time')
        end = time.time()
        model.train()
        if not args.no_progress:
            # p_bar = tqdm(range(args.eval_step),
            #              disable=args.local_rank not in [-1, 0])
            pass
        for batch_idx in range(args.eval_step):
            try:
                data_x = labeled_iter.next()
            except Exception:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                data_x = labeled_iter.next()

            try:
                data_u = unlabeled_iter.next()
            except Exception:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                data_u = unlabeled_iter.next()

            meter_manager.data_time.update(time.time() - end)
            # calculate loss
            loss_dict = model_trainer.compute_loss(
                data_x=data_x,
                data_u=data_u,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                iter=batch_idx,
                ema_model=ema_model,
                task_specific_info=task_specific_info,
                SCALER=SCALER if SCALER is not None else None)

            # update logger
            meter_manager.try_register_and_update(loss_dict)

            # step
            if SCALER is not None:
                SCALER.step(optimizer)
            else:
                optimizer.step()
            scheduler.step()

            # Updates the scale for next iteration
            if SCALER is not None:
                SCALER.update()

            # update ema if needed
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            meter_manager.batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress and args.local_rank in [-1, 0]:
                if batch_idx % cfg.log.interval == 0:
                    meter_desc = meter_manager.get_desc()
                    logger.info(
                        "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f} {desc}"
                        .format(epoch=epoch + 1,
                                epochs=args.epochs,
                                batch=batch_idx + 1,
                                iter=args.eval_step,
                                lr=scheduler.get_last_lr()[0],
                                desc=meter_desc))
                    # p_bar.update()

        # if not args.no_progress:
        #     p_bar.close()

        if args.local_rank in [-1, 0]:
            meter_manager.add_to_writer(args.writer, epoch, prefix="train/")

            # add test info
            test_model, test_prefix = get_test_model(ema_model, model,
                                                     args.use_ema)
            test_loss, test_acc = test(args, test_loader, test_model, epoch)
            args.writer.add_scalar('test/1.{}_acc'.format(test_prefix),
                                   test_acc, epoch)
            args.writer.add_scalar('test/2.{}_loss'.format(test_prefix),
                                   test_loss, epoch)
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            # add second evaluatino writer needed
            if cfg.get("evaluation", False) and cfg.evaluation.eval_both:
                test_model_second, test_prefix_second = get_test_model(
                    ema_model, model, not args.use_ema)
                test_loss_second, test_acc_second = test(
                    args, test_loader, test_model_second, epoch)
                args.writer.add_scalar(
                    'test/1.{}_acc'.format(test_prefix_second),
                    test_acc_second, epoch)
                args.writer.add_scalar(
                    'test/2.{}_loss'.format(test_prefix_second),
                    test_loss_second, epoch)

            # save model
            if epoch % cfg.ckpt.interval == 0:
                save_ckpt_dict(args, model, ema_model, epoch, test_acc,
                               optimizer, scheduler, task_specific_info,
                               is_best, best_acc)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

    # save last ckpt
    if args.local_rank in [-1, 0]:
        save_ckpt_dict(args, model, ema_model, epoch, test_acc, optimizer,
                       scheduler, task_specific_info, is_best, best_acc)

    if args.local_rank in [-1, 0]:
        args.writer.close()


#test/valiadate step


def test(args, test_loader, model, epoch):
    """ Test function for model and loader
        when the model is ema model, will test the ema model
        when the model is model, will test the regular model
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader, disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. \
                    Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. "
                    .format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                    ))
        if not args.no_progress:
            test_loader.close()

    logger.info("Epoch {} top-1 acc: {:.2f}".format(epoch, top1.avg))
    logger.info("Epoch {} top-5 acc: {:.2f}".format(epoch, top5.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
