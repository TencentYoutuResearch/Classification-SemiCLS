""" The Code is under Tencent Youtu Public Rule
config for training
althorithsm: CoMatch
datastet: STL-10
backbone: resnet18
batch per GPU: 64
num of GPUs: 1

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
        lambda_c=5.0,
        loss_x=dict(type='cross_entropy', reduction='mean')))
num_classes = 10
model = dict(
    type='resnet18',
    low_dim=64,
    num_class=10,
    proj=True,
    width=1,
    in_channel=3)
stl10_mean = (0.4914, 0.4822, 0.4465)
stl10_std = (0.2471, 0.2435, 0.2616)
data = dict(
    type='STL10SSL',
    folds=1,
    num_workers=4,
    num_classes=10,
    batch_size=64,
    mu=7,
    root='./data/stl10',
    lpipelines=[[{
        'type': 'RandomHorizontalFlip',
        'p': 0.5
    }, {
        'type': 'RandomCrop',
        'size': 96,
        'padding': 12,
        'padding_mode': 'reflect'
    }, {
        'type': 'ToTensor'
    }, {
        'type': 'Normalize',
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2471, 0.2435, 0.2616)
    }]],
    upipelinse=[[{
        'type': 'RandomHorizontalFlip',
        'p': 0.5
    }, {
        'type': 'RandomCrop',
        'size': 96,
        'padding': 12,
        'padding_mode': 'reflect'
    }, {
        'type': 'ToTensor'
    }, {
        'type': 'Normalize',
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2471, 0.2435, 0.2616)
    }],
                [{
                    'type': 'RandomHorizontalFlip',
                    'p': 0.5
                }, {
                    'type': 'RandomCrop',
                    'size': 96,
                    'padding': 12,
                    'padding_mode': 'reflect'
                }, {
                    'type': 'RandAugmentMC',
                    'n': 2,
                    'm': 10
                }, {
                    'type': 'ToTensor'
                }, {
                    'type': 'Normalize',
                    'mean': (0.4914, 0.4822, 0.4465),
                    'std': (0.2471, 0.2435, 0.2616)
                }],
                [{
                    'type': 'RandomResizedCrop',
                    'size': 96,
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
                    'mean': (0.4914, 0.4822, 0.4465),
                    'std': (0.2471, 0.2435, 0.2616)
                }]],
    vpipeline=[
        dict(type='ToTensor'),
        dict(
            type='Normalize',
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2471, 0.2435, 0.2616))
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
