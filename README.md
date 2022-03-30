# SemiCLS
This is the official implementation in PyTorch for cvpr2022 paper [Class-Aware Contrastive Semi-Supervised Learning](https://arxiv.org/abs/2203.02261) and also a semi-supervised learning toolbox based on mmcv.

<!-- # FixMatch
The code is changed from https://github.com/kekmodel/FixMatch-pytorch -->
**Supported algorithms**
- Supervised baseline
- FixMatch (NeurIPS 2020)[1]
- CoMatch (ICCV 2021)[2]
- FixMatch+CCSSL(CVPR 2022)[3]
- CoMatch+CCSSL

**Supported dataset**
- CIFAR10
- CIFAR100
- STL-10
- Customized dataset (e.g.,semi-iNat-2021)


## Results

### In-distribution datasets
|Method |       |CIFAR100  |       |       |CIFAR10|       |STL10  |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|       |400|2500|10000|40|250|4000|       |
|CoMatch|58.11±2.34|71.63±0.35|79.14±0.36|93.09±1.39|95.09±0.33|95.44±0.20|79.80±0.38|
|FixMatch|51.15±1.75|71.71±0.11|77.40±0.12|86.19±3.37|94.93±0.65|95.74±0.05|65.38±0.42|
|CCSSL(FixMatch)|61.19±1.65|75.7±0.63|80.68±0.16|90.83±2.78|94.86±0.55|95.54±0.20|80.01±1.39|

### Out-of-distribution datasets
| Method | Semi-iNat2021 | Semi-iNat2021 | 
| :---: | :---: | :---: |  
|| From Scratch | From MoCo Pretrain |  
| Supervised | 19.09 | 34.96|
| FixMatch |21.41  | 40.3 |
| FxiMatch+CCSSL |31.21 | 41.28 |
| CoMatch | 20.94 | 38.94 |
| CoMatch+CCSSL | 24.12 | 39.85 |


## Usage
### Install
Clone this repo to your machine and install dependencies:  
We use torch==1.6.0 and torchvision==0.12.0 for CUDA 10.1  
You may have to adapt for your own CUDA and install corresponding mmcv-full version. (Make sure your mmcv-full version is later than 1.3.2)
<!-- mmcv==1.4.6
mmcv_full==1.3.2 -->
or you can just:
```
pip install -r requirements.txt
```
### Train
1. **Env setup**  
  Set up your env with command below
```
export PYTHONPATH=$PYTHONPATH:`pwd`
```
2. **prepare datasets**  
   Organize your datasets as the following form:
```
data
└── CIFAR
│   └── cifar-10-batches-py # cifar10
│   └── cifar-100-python # cifar100
├── stl10
│   └── stl10_binary
└── semi-inat2021
│   ├── annotation_v2.json
│   ├── l_train
│   │   ├──anno.txt
│   │   └──l_train
│   │   │   ├──0
│   │   │   ├──1
│   │   │   │  └──0.jpg
│   │   │      ....
│   ├── u_train
│   │   ├──anno.txt
│   │   └──u_train
│   ├── val
│   │   ├──anno.txt
│   │   └──val
  ```

*Note: anno.txt contains data path and label(if have) for each image*, e.g.:

```
# prepare for semi-inat 2021, will print three txt path needed in config,
# like in configs/ccssl/fixmatchccssl_exp512_cifar100_wres_x8_b4x16_l2500_soft.py
python3 tools/data/prepare_semi_inat.py ./data/semi-inat2021

# anno.txt under l_train
your/dataste/semi-inat-2021/l_train/l_train/1/0.jpg 1

# anno.txt under u_train
your/dataste/semi-inat-2021/l_train/u_train/xxxxx.jpg
```


3. Now you can run the experiments for different SSL althorithms by modifying configs as you need.  
Code examples are as follow:

 ```
 ## Single-GPU
 # to train the model by 40 labeled data of CIFAR-10 dataset by FixMatch:
 python3 train_semi.py --cfg ./configs/fixmatch/fm_cifar10_wres_b1x64_l250.py --out your/output/path   --seed 5 --gpu-id 0

## Multi-GPU
# to train the model by CIFAR100 dataset by FixMatch+CCSSL with 4GPUs:
 python3 -m torch.distributed.launch --nproc_per_node 4 train_semi.py --cfg ./configs/ccssl/fixmatchccssl_exp512_cifar100_wres_x8_b4x16_l2500_soft.py --out /your/output/path --use_BN True  --seed 5

# to train the model by Semi-iNat2021 dataset by FixMatch+CCSSL with 4GPUs:
 python3 -m torch.distributed.launch --nproc_per_node 4 train_semi.py --cfg ./configs/ccssl/fixmatchccssl_exp512_seminat_b4x16_soft06_push09_mu7_lc2.py --out /your/output/path --use_BN True  --seed 5
 ```
## Customization
1. If you want to write your own SSL althorithm, e.g., your_SSL, you need to wirte it in `trainer/your_SSL.py` and remember to register it in `trainer/builder`.py
2. If you want to add other `backbones|loss functions|data transforms` you need, please write it under `models|loss|dataset\transforms|` and also remember to register it in the `builder.py` under the same folder.
3. For customized datasets, we provide two data options in the config files :`"MyDataset"` for dataset in the form of imagefolder and `"TxtDatasetSSL"` for dataset with txt annotations.

## BibTex Citation
If you think our work or this code is helpful for your research, please cite its arxiv version using the following BibTex (we will update its CVPR 2022 version later):
```
@article{yang2022class,
  title={Class-Aware Contrastive Semi-Supervised Learning},
  author={Yang, Fan and Wu, Kai and Zhang, Shuyi and Jiang, Guannan and Liu, Yong and Zheng, Feng and Zhang, Wei and Wang, Chengjie and Zeng, Long},
  journal={arXiv preprint arXiv:2203.02261},
  year={2022}
}
```

## Reference
[1] Kihyuk Sohn, David Berthelot, Nicholas Carlini, Zizhao Zhang, Han Zhang, Colin A Raf-fel, Ekin Dogus Cubuk, Alexey Kurakin, and Chun-Liang Li.  Fixmatch:  Simplifying semi-supervised learning with consistency and confidence.NeurIPS, 33, 2020.  
[2] Li, Junnan, Caiming Xiong, and Steven CH Hoi. "Comatch: Semi-supervised learning with contrastive graph regularization." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.  
[3] Yang, Fan, et al. "Class-Aware Contrastive Semi-Supervised Learning." arXiv preprint arXiv:2203.02261 (2022).

## Contact us
Feel free to open an issue, submit a merge request or send an email us  
Fan Yang: fan-yang20@mails.tsinghua.edu.cn  
Kai Wu: lloydwu@tencent.com
