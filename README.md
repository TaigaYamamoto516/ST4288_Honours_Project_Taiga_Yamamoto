# ST4288_Taiga_Yamamoto

This repository contains code used for "XXXX".

## MMD GAN

The code of MMD GAN is based on [MMD-GAN: Towards Deeper Understanding of Moment Matching Network](https://github.com/OctoberChang/MMD-GAN).

You can use a file "./MMD-GAN-PyTorch/MMD-GAN-PyTorch.ipynb" for the experiment.

```
./mmd_gan [OPTIONS]
OPTIONS:
    --dataset DATASET: type of dataset (mnist/cifar10/celeba/lsun)
    --dataroot DATAROOT: path to dataset
    --glrやdlr, g_iterationなどを追加しても良いかも。

```

To conduct the experiment, run
```
    chmod +x run_exp_LM.sh
    ./run_exp.sh [mnist/cifar10/celeba/lsun][XXXXX]

```

Diiferent kinds of codes

## WGAN

The code of WGAN is based on [the Github page](https://github.com/Zeleni9/pytorch-wgan).

You can use a file "./pytorch-wgan/wgan-pytorch.ipynb" to conduct the experiments.

To run WGAN-GP

```
python main.py --model WGAN-GP \
               --is_train True \
               --download True \
               --dataroot cifar \
               --dataset cifar \
               --generator_iters 15000 \
               --cuda True \
               --batch_size 64 \
               --optimizer [RMSprop/Adam/NAdam] \
               --lm [Original/IKSA]
```

Although there are various algorithms: DCGAN, GAN, WGAN, WGAN-GP, WGAN-CP, only WGAN-GP works for Landscape Modification.

## Datasets

Put datasets in "./MMD-GAN-PyTorch/data".

## Calculation of FID and Inception Score

You can use a package "pytorch-gan-metrics" for the calculation of FID and Inception Score.
Please refer to [the Github page](https://github.com/w86763777/pytorch-gan-metrics).

Also, you can a file "Calculation_FID_IS.ipynb" to calculate the scores.

```
The files required for the calculation of the evaluation scores are in the folder below:
    - ./MMD-GAN-PyTorch/Generarted_png/
```

## Optimizers with Landscape Modification

The optimizers with Landscape Modification are based on the paper "Landscape Modification in Machine Learning Optimization".

Please refer to [the Github page](https://github.com/IoanaTodea22/LandscapeModification.git).

## More Info

For any questions and comments, please send your email to taigayamamoto@u.nus.edu or taiga2002516@gmail.com
