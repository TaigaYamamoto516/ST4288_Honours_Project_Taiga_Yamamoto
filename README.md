# ST4288_Taiga_Yamamoto

This repository contains code used for "XXXX" for the ST4288 project.

## MMD GAN

The code of MMD GAN is based on [MMD-GAN: Towards Deeper Understanding of Moment Matching Network](https://github.com/OctoberChang/MMD-GAN).

```
./mmd_gan [OPTIONS]
OPTIONS:
    --dataset DATASET: type of dataset (mnist/cifar10/celeba/lsun)
    --dataroot DATAROOT: path to dataset
    --workers WORKERS: number of threads to load data
    --batch_size BATCH_SIZE: batch size for training
    --image_size IMAGE_SIZE: image size of dataset
    --nc NC: number of channels in images
    --nz NZ: hidden dimension in z and codespace
    --max_iter MAX_ITER: max iteration for training
    --lr LR: learning rate (default 5e-5)
    --gpu_device GPU_DEVICE: gpu id (default 0)
    --netG NETG: path to generator model
    --netD NETD: path to discriminator model
    --Diters DITERS: number of updates for discriminator per one generator update
    --experiment EXPERIMENT: output directory of sampled images
```



## WGAN

The code of WGAN is based on [the Github page](https://github.com/Zeleni9/pytorch-wgan).

```
python main.py --model DCGAN \
               --is_train True \
               --download True \
               --dataroot datasets/fashion-mnist \
               --dataset fashion-mnist \
               --epochs 30 \
               --cuda True \
               --batch_size 64
```



## Datasets



## Calculation of FID and Inception Score

You can use a package "pytorch-gan-metrics" for the calculation of FID and Inception Score.
Please refer to [the Github page](https://github.com/w86763777/pytorch-gan-metrics).


Also, you can a file "XXX" to calculate the scores

The files required for the calculation of the evaluation scores are below:
    - "XXXX" for CIFAR-10
    - "XXXX" for LSUN Bedroom
    - "XXXX" for CelebA

## Optimizers with Landscape Modification

The optimizers with Landscape Modification are based on the paper "Landscape Modification in Machine Learning Optimization" by Ioana Todea.
Please refer to [the Github page](https://github.com/IoanaTodea22/LandscapeModification.git)
