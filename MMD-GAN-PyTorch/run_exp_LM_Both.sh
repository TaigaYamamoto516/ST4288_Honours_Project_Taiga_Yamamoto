#!/bin/bash

BS=64
GPU_ID=0
MAX_ITER=210
DATA_PATH=./data


#Choose datasets
if [ $1 == 'mnist' ]; then
    DATASET=mnist
    DATAROOT=${DATA_PATH}/mnist
    ISIZE=32
    NC=3
    NZ=10

elif [ $1 == 'cifar10' ]; then
    DATASET=cifar10
    DATAROOT=${DATA_PATH}/cifar10
    ISIZE=32
    NC=3
    NZ=128
     
elif [ $1 == 'celeba' ]; then
    DATASET=celeba
    DATAROOT=${DATA_PATH}/celeba
    ISIZE=64
    NC=3
    NZ=64

elif [ $1 == 'lsun' ]; then
    DATASET=lsun
    DATAROOT=${DATA_PATH}/lsun
    ISIZE=64
    NC=3
    NZ=128

else
    echo "unknown dataset [mnist /cifar10 / celeba / lsun]"
    exit
fi

if [ $2 == 'Adagrad' ]; then
    OPTIMIZER=Adagrad
  if [ $3 == 'Original' ]; then
    MODEL=Original
  elif [ $3=='LM' ]; then
    MODEL=IKSA
  else
    echo 'unknown model [Original / LM]'
    exit
  fi

elif [ $2 == 'Adam' ]; then
    OPTIMIZER=Adam
  if [ $3 == 'Original' ]; then
    MODEL=Original
  elif [ $3 == 'LM' ]; then
    MODEL=IKSA
  else
    echo 'unknown model [Original / LM]'
    exit
  fi

elif [ $2 == 'NAdam' ]; then
    OPTIMIZER=NAdam
  if [ $3 == 'Original' ]; then
    MODEL=Original
  elif [ $3 == 'LM' ]; then
    MODEL=IKSA
  else
    echo 'unknown model [Original / LM]'
    exit
  fi

elif [ $2 == 'RMSprop' ]; then
    OPTIMIZER=RMSprop
  if [ $3 == 'Original' ]; then
    MODEL=Original
  elif [ $3 == 'LM' ]; then
    MODEL=IKSA
  else
    echo 'unknown model [Original / LM]'
    exit
  fi

elif [ $2 == 'SGD' ]; then
  OPTIMIZER=SGD
  if [ $3 == 'Original' ]; then
    MODEL=Original
  elif [ $3 == 'LM' ]; then
    MODEL=IKSA
  else
    echo 'unknown model [Original / LM]'
    exit
  fi

else
  echo 'unknown optimizer [Adagrad / NAdam / Adam / RMSprop / SGD]'
  exit
fi

#Check if it is　expression is right
EXP_FILE="./Generated_Images/${DATASET}/${OPTIMIZER}/${DATASET}_${OPTIMIZER}_${MODEL}_Both_i=${MAX_ITER}_mmd-gan"
LOG_FILE="./log/${DATASET}/${OPTIMIZER}/${DATASET}_${OPTIMIZER}_${MODEL}_Both_i=${MAX_ITER}_mmd-gan.log"
DATA_FILE="./Data_Results/${DATASET}/${OPTIMIZER}/${DATASET}_${OPTIMIZER}_${MODEL}_Both_i=${MAX_ITER}_mmd-gan"

cmd="stdbuf -o L python mmd_gan_LM_Both.py --dataset ${DATASET} --dataroot ${DATAROOT} --optimizer ${OPTIMIZER} --model ${MODEL} --batch_size ${BS} --image_size ${ISIZE} --nc ${NC}  --nz ${NZ} --max_iter ${MAX_ITER} --gpu_device ${GPU_ID} --data_file ${DATA_FILE} --experiment ${EXP_FILE} | tee ${LOG_FILE}"

echo $cmd
eval $cmd

#$1: DATASETS ['mnist', 'cifar10', 'celeba', lsun']
#$2: OPTIMIZER ['Adagrad', 'Adam', 'NAdam', 'RMSprop', 'SGD']
#$3: MODEL ['Original', 'LM']
