""" The Code is under Tencent Youtu Public Rule
config for training
althorithsm: CoMatch
datastet: CIFAR100
backbone: wideresnet28x8
batch per GPU: 16
num of GPUs: 4
num of labeled data: 2500

this file is organized in the form of
|-train
    |-training related main options
    |-trainer()：SSL althorithsm
|-model
|-data
    |-type: CIFAR10/CIFAR100/STL-10/customized dataset
    ｜-num_labeled：num of labeled data
    |-other data related options
    |-transforms
|-other options

"""
train = dict(
    eval_step=1024,
    total_steps=524288,
    trainer=dict(
        type='CoMatch',
        threshold=0.95,
        queue_batch=5,
        contrast_threshold=0.8,
        da_len=32,
        T=0.2,
        alpha=0.9,
        lambda_u=1.0,
        lambda_c=1.0,
        loss_x=dict(type='cross_entropy', reduction='mean')))
num_classes = 100
model = dict(
    type='wideresnet',
    depth=28,
    widen_factor=8,
    dropout=0,
    num_classes=100,
    proj=True,
    low_dim=64)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
data = dict(
    type='CIFAR100SSL',
    num_workers=4,
    num_labeled=2500,
    num_classes=100,
    batch_size=16,
    expand_labels=False,
    mu=7,
    root='./data/CIFAR',
    lpipelines=[[{
        'type': 'RandomHorizontalFlip',
        'p': 0.5
    }, {
        'type': 'RandomCrop',
        'size': 32,
        'padding': 4,
        'padding_mode': 'reflect'
    }, {
        'type': 'ToTensor'
    }, {
        'type': 'Normalize',
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761)
    }]],
    upipelinse=[[{
        'type': 'RandomHorizontalFlip',
        'p': 0.5
    }, {
        'type': 'RandomCrop',
        'size': 32,
        'padding': 4,
        'padding_mode': 'reflect'
    }, {
        'type': 'ToTensor'
    }, {
        'type': 'Normalize',
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761)
    }],
                [{
                    'type': 'RandomHorizontalFlip',
                    'p': 0.5
                }, {
                    'type': 'RandomCrop',
                    'size': 32,
                    'padding': 4,
                    'padding_mode': 'reflect'
                }, {
                    'type': 'RandAugmentMC',
                    'n': 2,
                    'm': 10
                }, {
                    'type': 'ToTensor'
                }, {
                    'type': 'Normalize',
                    'mean': (0.5071, 0.4867, 0.4408),
                    'std': (0.2675, 0.2565, 0.2761)
                }],
                [{
                    'type': 'RandomResizedCrop',
                    'size': 32,
                    'scale': (0.2, 1.0)
                }, {
                    'type': 'RandomHorizontalFlip',
                    'p': 0.5
                }, {
                    'type':
                    'RandomApply',
                    'transforms': [{
                        'type': 'ColorJitter',
                        'brightness': 0.4,
                        'contrast': 0.4,
                        'saturation': 0.4,
                        'hue': 0.1
                    }],
                    'p':
                    0.8
                }, {
                    'type': 'RandomGrayscale',
                    'p': 0.2
                }, {
                    'type': 'ToTensor'
                }, {
                    'type': 'Normalize',
                    'mean': (0.5071, 0.4867, 0.4408),
                    'std': (0.2675, 0.2565, 0.2761)
                }]],
    vpipeline=[
        dict(type='ToTensor'),
        dict(
            type='Normalize',
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761))
    ],
    eval_step=1024)
scheduler = dict(
    type='cosine_schedule_with_warmup',
    num_warmup_steps=0,
    num_training_steps=524288)
ema = dict(use=True, pseudo_with_ema=False, decay=0.999)
amp = dict(use=False, opt_level='O1')
log = dict(interval=1)
ckpt = dict(interval=1)
optimizer = dict(
    type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0005, nesterov=True)
resume = 'True'
