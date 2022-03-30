""" The Code is under Tencent Youtu Public Rule
config for training
althorithsm: FixMatch
datastet: CIFAR10
backbone: wideresnet28x2
batch per GPU: 64
num of GPUs: 1
num of labeled data: 250

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
train = dict(eval_step=1024,
             total_steps=2**20,
             trainer=dict(type="FixMatch",
                          threshold=0.95,
                          T=1.,
                          lambda_u=1.,
                          loss_x=dict(
                              type="cross_entropy",
                              reduction="mean"),
                          loss_u=dict(
                              type="cross_entropy",
                              reduction="none"),
                          ))
num_classes = 10
# seed = 1

model = dict(
     type="wideresnet",
     depth=28,
     widen_factor=2,
     dropout=0,
     num_classes=num_classes,
)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

data = dict(
    # CIFAR10SSL, CIFAR100SSL
    type="CIFAR10SSL",
    num_workers=4,
    num_labeled=250,
    num_classes=num_classes,
    batch_size=64,
    expand_labels=False,
    mu=7,

    root="./data/CIFAR",
    lpipelines=[[
        dict(type="RandomHorizontalFlip"),
        dict(type="RandomCrop",
             size=32,
             padding=int(32 * 0.125),
             padding_mode='reflect'),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=cifar10_mean, std=cifar10_std)
    ]],
    upipelinse=[[
        dict(type="RandomHorizontalFlip"),
        dict(type="RandomCrop",
             size=32,
             padding=int(32 * 0.125),
             padding_mode='reflect'),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=cifar10_mean, std=cifar10_std)],[
        dict(type="RandomHorizontalFlip"),
        dict(type="RandomCrop",
             size=32,
             padding=int(32 * 0.125),
             padding_mode='reflect'),
        dict(type="RandAugmentMC", n=2, m=10),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=cifar10_mean, std=cifar10_std)]],
    vpipeline=[
        dict(type="ToTensor"),
        dict(type="Normalize", mean=cifar10_mean, std=cifar10_std)
    ])

scheduler = dict(
    type='cosine_schedule_with_warmup',
    num_warmup_steps=0,
    num_training_steps=train['total_steps']
)

ema = dict(use=True, pseudo_with_ema=False, decay=0.999)
#apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
#"See details at https://nvidia.github.io/apex/amp.html
amp = dict(use=False, opt_level="O1")

log = dict(interval=1)
ckpt = dict(interval=1000)

# optimizer
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.001, nesterov=True)
