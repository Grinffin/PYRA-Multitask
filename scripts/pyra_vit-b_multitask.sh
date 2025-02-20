#!/usr/bin/env bash

set -x

currenttime=`date "+%Y%m%d_%H%M%S"`

CONFIG="ViT-B_prompt_lora_8"
CONFIG_DIR="experiments/LoRA/"${CONFIG}".yaml"
CKPT="weights/ViT-B_16.npz"
WEIGHT_DECAY=0.0001

device=$1
merge_schedule=$3                       # "low" "high"
slice_choice=$2
if [ "${slice_choice}" = "1" ]; then
    DATASETS=(cifar100 caltech101 dtd oxford_flowers102 oxford_pet)
    if [ "${merge_schedule}" = "low" ]; then
        PYRA_LR=(1e-5 1e-3 3e-4 1e-5 3e-4)
    else
        PYRA_LR=(3e-5 3e-5 3e-4 1e-5 1e-6)
    fi
fi
if [ "${slice_choice}" = "2" ]; then
    DATASETS=(eurosat resisc45 clevr_count clevr_dist)
    if [ "${merge_schedule}" = "low" ]; then
        PYRA_LR=(1e-2 1e-3 1e-4 1e-5)
    else
        PYRA_LR=(3e-4 1e-4 3e-6 1e-6)
    fi 
fi
if [ "${slice_choice}" = "3" ]; then
    DATASETS=(dmlab kitti dsprites_loc dsprites_ori smallnorb_azi)
    if [ "${merge_schedule}" = "low" ]; then
        PYRA_LR=(1e-2 3e-6 3e-4 3e-4 1e-2)
    else
        PYRA_LR=(3e-5 1e-4 1e-6 1e-5 3e-5)
    fi
fi
if [ "${slice_choice}" = "4" ]; then
    DATASETS=(svhn sun397 patch_camelyon diabetic_retinopathy smallnorb_ele)
    if [ "${merge_schedule}" = "low" ]; then
        PYRA_LR=(3e-2 3e-6 1e-3 1e-2 3e-4)
    else
        PYRA_LR=(1e-3 1e-6 1e-4 1e-3 3e-4)
    fi
fi
if [ "${slice_choice}" = "5" ]; then
    DATASETS=(natural)
    if [ "${merge_schedule}" = "low" ]; then
        PYRA_LR=(3e-2 3e-6 1e-3 1e-2 3e-4)
        PYRA_LR=(1e-4)
    else
        PYRA_LR=(1e-3 1e-6 1e-4 1e-3 3e-4)
        PYRA_LR=(1e-4)
    fi
fi


i=0
    for DATASET in "${DATASETS[@]}"
    do
        pyra_lr=${PYRA_LR[i]}
        i=$((i+1))
        for LR in 0.001
        do
            LOG_DIR=outputs/240910_${CONFIG}_compress_${merge_schedule}_PYRA_test_RUN_${RUN}
            
            TARGET_DIR=${LOG_DIR}/${DATASET}_lr-${LR}_wd-${WEIGHT_DECAY}_pyra_lr-${pyra_lr}
            if [ ! -d ${TARGET_DIR} ]
            then
                mkdir -p ${LOG_DIR}
                mkdir -p ${TARGET_DIR}
            else
                echo "Dir already exists, skipping ${TARGET_DIR}"
                #continue
            fi
            CUDA_VISIBLE_DEVICES=${device} python train.py --data-path=./data/vtab-1k/${DATASET} --data-set=${DATASET}\
                    --cfg=${CONFIG_DIR} --resume=${CKPT} --output_dir=${TARGET_DIR}\
                    --batch-size=32 --lr=${LR} --epochs=100 --weight-decay=${WEIGHT_DECAY}\
                    --no_aug --mixup=0 --cutmix=0 --direct_resize --smoothing=0 --save_ckpt\
                    --token_merging --merging_schedule=${merge_schedule}\
                    --pyra --separate_lr_for_pyra --pyra_lr=${pyra_lr}\
            2>&1 | tee -a ${LOG_DIR}/${DATASET}_lr-${LR}_wd-${WEIGHT_DECAY}_pyra_lr-${pyra_lr}.log > /dev/null
             #    --pyra --separate_lr_for_pyra --pyra_lr=${pyra_lr}\     
             # --token_merging --merging_schedule=${merge_schedule}\           
        done
    done
