#!/usr/bin/env bash

set -x

currenttime=`date "+%Y%m%d_%H%M%S"`

CONFIG="ViT-B_prompt_lora_8"
CONFIG_DIR="experiments/LoRA/"${CONFIG}".yaml"
CKPT=$4 # path to the checkpoint (best_checkpoint.pth)
WEIGHT_DECAY=0.0001

device=$1
DATASET=$2
merge_schedule=$3                       # "low" "high"
offset=$5

# for RUN in {1..5}
# do
# i=0
    # for DATASET in "${DATASETS[@]}"
    # do
        pyra_lr=${PYRA_LR[i]}
        i=$((i+1))
        for LR in 0.001
        do
            # evaluate PYRA
            # LOG_DIR=outputs/240907_${CONFIG}_compress_${merge_schedule}_PYRA_RUN_${RUN}_eval

            # evaluate ToMe 
            #LOG_DIR=outputs/240910_${CONFIG}_compress_${merge_schedule}_eval
            LOG_DIR=outputs/tmp/${DATASET}_pyra+
            
            TARGET_DIR=${LOG_DIR}/${DATASET}_lr-${LR}_wd-${WEIGHT_DECAY}_pyra_lr-${pyra_lr}

            if [ ! -d ${TARGET_DIR} ]
            then
                mkdir -p ${LOG_DIR}
                mkdir -p ${TARGET_DIR}
            else
                echo "Dir already exists, skipping ${TARGET_DIR}"
                #continue
            fi

            # evaluate PYRA
            # CUDA_VISIBLE_DEVICES=${device} python train.py --data-path=./data/vtab-1k/${DATASET} --data-set=${DATASET}\
            #         --cfg=${CONFIG_DIR} --resume=${CKPT} --output_dir=${TARGET_DIR}\
            #         --batch-size=32 --lr=${LR} --epochs=100 --weight-decay=${WEIGHT_DECAY}\
            #         --no_aug --mixup=0 --cutmix=0 --direct_resize --smoothing=0 --save_ckpt\
            #         --token_merging --merging_schedule=${merge_schedule}\
            #         --pyra --separate_lr_for_pyra --pyra_lr=${pyra_lr}\
            # 2>&1 | tee -a ${LOG_DIR}/${DATASET}_lr-${LR}_wd-${WEIGHT_DECAY}.log > /dev/null

            # evaluate ToMe 
            CUDA_VISIBLE_DEVICES=${device} python3 train.py --data-path=./data/vtab-1k/${DATASET} --data-set=${DATASET}\
                    --cfg=${CONFIG_DIR} --resume=${CKPT} --output_dir=${TARGET_DIR}\
                    --batch-size=32 --lr=${LR} --epochs=100 --weight-decay=${WEIGHT_DECAY}\
                    --no_aug --mixup=0 --cutmix=0 --direct_resize --smoothing=0 --save_ckpt --eval\
                    --offset=${offset}\
                    --token_merging --merging_schedule=${merge_schedule}\
                    --pyra\
                  
            2>&1 | tee -a ${LOG_DIR}/${DATASET}_lr-${LR}_wd-${WEIGHT_DECAY}.log > /dev/null 
        done
    # done
# done