
# Towards a Unified Theoretical Understanding of Non-contrastive Learning via Rank Differential Mechanism

This is a PyTorch implementation of ICLR 2023 paper [Towards a Unified Theoretical Understanding of Non-contrastive Learning via Rank Differential Mechanism](https://openreview.net/forum?id=cIbjyd2Vcy) authored by Zhijian Zhuo, [Yifei Wang](yifeiwang77.com), Jinwen Ma, [Yisen Wang](https://yisenwang.github.io/).

## Abstract

Recently, a variety of methods under the name of non-contrastive learning (like BYOL, SimSiam, SwAV, DINO) show that when equipped with some asymmetric architectural designs, aligning positive pairs alone is sufficient to attain good performance in self-supervised visual learning. Despite some understandings of some specific modules (like the predictor in BYOL), there is yet no unified theoretical understanding of how these seemingly different asymmetric designs can all avoid feature collapse, particularly considering  methods that also work without the predictor (like DINO). In this work, we propose a unified theoretical understanding for existing variants of non-contrastive learning. Our theory named Rank Differential Mechanism (RDM) shows that all these asymmetric designs create a consistent rank difference in their dual-branch output features. This rank difference will provably lead to an improvement of effective dimensionality and alleviate either complete or dimensional feature collapse. Different from previous theories, our RDM theory is applicable to different asymmetric designs (with and without the predictor), and thus can serve as a unified understanding of existing non-contrastive learning methods. Besides, our RDM theory also provides practical guidelines for designing many new non-contrastive variants. We show that these variants indeed achieve comparable performance to existing methods on benchmark datasets, and some of them even outperform the baselines.

## Quick start

### CIFAR-10/100 and ImageNet-100
Methods are based on SimSiam and BYOL in [solo-learn](https://github.com/vturrisi/solo-learn) with minor modifications and we follow all the default configurations of [solo-learn](https://github.com/vturrisi/solo-learn). 

For pretraining the backbone on CIFAR-10 and 100, follow bash files: `scripts/pretrain/cifar/rdmsimsiam.sh`, `scripts/pretrain/cifar/rdmbyol.sh`.  To train RDM_SimSiam and RDM_BYOL on ImageNet-100, follow bash files : `scripts/pretrain/imagenet-100/rdmsimsiam.sh` and `scripts/pretrain/cifar/rdmbyol.sh`.  The following are additional hyper-parameters `pred_type`, `pred_location` and `sigma`: 

- `pred_type` is the type of the predictor (`poly`, `log`, `log_1`, `log_2`); 
- `pred_location` is the location of the predictor (`online`,`target`); 
- `sigma` is the power of the power function.

For example, train RDM_SimSiam on CIFAR-10 with a target predictor $h(x)=x^{-1}$ for 400 epochs on a single GPU:
```
sh bash_files/pretrain/cifar/rdmsimsiam.sh cifar10 400 poly target -1
```

Train RDM_BYOL on ImageNet-100 with an online predictor $g(x)=x^{1}$ for 400 epochs on two GPUs:
```
sh bash_files/pretrain/imagenet-100/rdmbyol.sh 400 poly online 1
```



### ImageNet-1k

The code for RDM_SimSiam on ImageNet-1k is in `./imagenet`, which
is based on [the official code of SimSiam](https://github.com/facebookresearch/simsiam) with minor modifications and we follow all the default training and evaluation configurations. 

```
cd ./imagenet
```

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:


```
python main_simsiam.py \
  -a resnet50 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed \
  --world-size 1 --rank 0 \
  --fix-pred-lr \
  --pred-type poly \
  --sigma 1 \
  --pred-location target \
  [your imagenet-folder with train and val folders]

```

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:
```
python main_lincls.py \
  -a resnet50 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained [your checkpoint path]/checkpoint_0099.pth.tar \
  --lars \
  [your imagenet-folder with train and val folders]
```

## Citation
To cite our article, please cite:
```
@inproceedings{
    zhuo2023towards,
    title={Towards a Unified Theoretical Understanding of Non-contrastive Learning via Rank Differential Mechanism},
    author={Zhuo, Zhijian and Wang, Yifei and Ma, Jinwen and Wang, Yisen},
    booktitle={International Conference on Learning Representations},
    year={2023},
}
```

## Acknowledgement
Our code is based on the Self-Supervised learning library [solo-learn](https://github.com/vturrisi/solo-learn) and [SimSiam](https://github.com/facebookresearch/simsiam).
