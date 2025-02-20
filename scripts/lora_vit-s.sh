#!/usr/bin/env bash

set -x

currenttime=`date "+%Y%m%d_%H%M%S"`

CONFIG="ViT-S_prompt_lora_4"
CONFIG_DIR="experiments/LoRA/"${CONFIG}".yaml"
CKPT="weights/imagenet21k_ViT-S_16.npz"
WEIGHT_DECAY=0.0001

device=$1
DATASETS=(cifar100 caltech101 dtd oxford_flowers102 svhn sun397 oxford_pet patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele)
RUN=$2
# for RUN in {1..5}
# do
# i=0
    for DATASET in "${DATASETS[@]}"
    do
        pyra_lr=${PYRA_LR[i]}
        i=$((i+1))
        for LR in 0.001
        do
            LOG_DIR=outputs/240924_${CONFIG}_plain_lora_batch_128_RUN_${RUN}
            
            TARGET_DIR=${LOG_DIR}/${DATASET}_lr-${LR}_wd-${WEIGHT_DECAY}
            if [ ! -d ${TARGET_DIR} ]
            then
                mkdir -p ${LOG_DIR}
                mkdir -p ${TARGET_DIR}
            else
                echo "Dir already exists, skipping ${TARGET_DIR}"
                continue
            fi
            CUDA_VISIBLE_DEVICES=${device} python train.py --data-path=./data/vtab-1k/${DATASET} --data-set=${DATASET}\
                    --cfg=${CONFIG_DIR} --resume=${CKPT} --output_dir=${TARGET_DIR}\
                    --batch-size=128 --lr=${LR} --epochs=100 --weight-decay=${WEIGHT_DECAY}\
                    --no_aug --mixup=0 --cutmix=0 --direct_resize --smoothing=0 --save_ckpt\
            2>&1 | tee -a ${LOG_DIR}/${DATASET}_lr-${LR}_wd-${WEIGHT_DECAY}.log > /dev/null 
        done
    done
# done