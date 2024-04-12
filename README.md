# ST4288_Honours_Project_Taiga_Yamamoto

This repository contains code used for [Optimizing Generative Adversarial Networks: The Impact of Landscape Modification on GAN Performance](https://github.com/TaigaYamamoto516/ST4288_Honours_Project_Taiga_Yamamoto/blob/main/ST4288_Honours_Thesis_Taiga_Yamamoto.pdf).

## MMD GAN

The code of MMD GAN is based on [MMD-GAN: Towards Deeper Understanding of Moment Matching Network](https://github.com/OctoberChang/MMD-GAN).

You can use the file [./MMD-GAN-PyTorch/MMD-GAN-PyTorch.ipynb" for the experiment](https://github.com/TaigaYamamoto516/ST4288_Honours_Project_Taiga_Yamamoto/blob/main/MMD-GAN-PyTorch/MMD-GAN-PyTorch.ipynb).

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
    --gpu_device GPU_DEVICE: gpu id (default 0)
    --netG NETG: path to generator model
    --netD NETD: path to discriminator model
    --Diters DITERS: number of updates for discriminator per one generator update
    --experiment EXPERIMENT: output directory of sampled images
    --optimizer: types of optimizers
    --model: LM or Original
    --LM_f: function for LM
    --data_file: where to store the experiment data
    --dlr: learning rate for discriminator (default 0.0001)
    --glr: learning rate for generator (default 0.0002)
    --g_max_iteration, iteration times for generator (default 25000)

For LM_f, dlr, glr, and g_max_iteration, please change the variables in the "util.py" file directly.

```

To conduct the experiment, run
```

    chmod +x run_exp_LM.sh
    ./run_exp.sh [datasets] [Optimizer] [Model: Original or LM]

```

For more details about running each model in the paper, please refer to the file [./MMD-GAN-PyTorch/MMD-GAN-PyTorch.ipynb](https://github.com/TaigaYamamoto516/ST4288_Honours_Project_Taiga_Yamamoto/blob/main/MMD-GAN-PyTorch/MMD-GAN-PyTorch.ipynb)'.

## WGAN-GP

The code of WGAN is based on [the Github page](https://github.com/Zeleni9/pytorch-wgan).

You can use the file [./pytorch-wgan/wgan-pytorch.ipynb" to conduct the experiments](https://github.com/TaigaYamamoto516/ST4288_Honours_Project_Taiga_Yamamoto/blob/main/pytorch-wgan/wgan-pytorch.ipynb).

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

Although the folder contains various algorithms: DCGAN, GAN, WGAN, WGAN-GP, and WGAN-CP, only WGAN-GP works for Landscape Modification.

## Datasets

Put datasets in "./MMD-GAN-PyTorch/data".

## Calculation of FID and Inception Score

You can use the package "pytorch-gan-metrics" to calculate the FID and Inception Score.

Please refer to [the Github page](https://github.com/w86763777/pytorch-gan-metrics).

Also, you can use the file [Calculation_FID_IS.ipynb" to calculate the scores](https://github.com/TaigaYamamoto516/ST4288_Honours_Project_Taiga_Yamamoto/blob/main/Calculation_FID_IS.ipynb).

Download the data required for the calculation of FID and IS from [here](https://drive.google.com/drive/folders/1UBdzl6GtNMwNQ5U-4ESlIer43tNjiGJC).


## Optimizers with Landscape Modification

The optimizers with Landscape Modification are based on the paper "Landscape Modification in Machine Learning Optimization".

Please refer to [the Github page](https://github.com/IoanaTodea22/LandscapeModification.git).

## More Info

For any questions and comments, please send your email to taigayamamoto@u.nus.edu or taiga2002516@gmail.com.
