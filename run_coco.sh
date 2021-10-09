#!/usr/bin/env bash

EXPNAME=$1
SAVEDIR=checkpoints/coco/${EXPNAME}
IMAGENET_PRETRAIN=/data/.pretrain_weights/ImageNetPretrained/MSRA/R-101.pkl                            # <-- change it to you path
IMAGENET_PRETRAIN_TORCH=/data/.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth  # <-- change it to you path


# ------------------------------- Base Pre-train ---------------------------------- #
python3 main.py --num-gpus 8 --config-file configs/coco/defrcn_det_r101_base.yaml     \
    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                         \
           OUTPUT_DIR ${SAVEDIR}/defrcn_det_r101_base


# ------------------------------ Model Preparation -------------------------------- #
python3 tools/model_surgery.py --dataset coco --method remove                         \
    --src-path ${SAVEDIR}/defrcn_det_r101_base/model_final.pth                        \
    --save-dir ${SAVEDIR}/defrcn_det_r101_base
BASE_WEIGHT=${SAVEDIR}/defrcn_det_r101_base/model_reset_remove.pth


# ------------------------------ Novel Fine-tuning -------------------------------- #
# --> 1. FSRW-like, i.e. run seed0 10 times (the FSOD results on coco in most papers)
for repeat_id in 0 1 2 3 4 5 6 7 8 9
do
    for shot in 1 2 3 5 10 30
    do
        for seed in 0
        do
            python3 tools/create_config.py --dataset coco14 --config_root configs/coco \
                --shot ${shot} --seed ${seed} --setting 'fsod'
            CONFIG_PATH=configs/coco/defrcn_fsod_r101_novel_${shot}shot_seed${seed}.yaml
            OUTPUT_DIR=${SAVEDIR}/defrcn_fsod_r101_novel/fsrw-like/${shot}shot_seed${seed}_repeat${repeat_id}
            python3 main.py --num-gpus 8 --config-file ${CONFIG_PATH}                  \
                --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}           \
                       TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}
            rm ${CONFIG_PATH}
            rm ${OUTPUT_DIR}/model_final.pth
        done
    done
done
python3 tools/extract_results.py --res-dir ${SAVEDIR}/defrcn_fsod_r101_novel/fsrw-like --shot-list 1 2 3 5 10 30  # surmarize all results


# ----------------------------- Model Preparation --------------------------------- #
python3 tools/model_surgery.py --dataset coco --method randinit                        \
    --src-path ${SAVEDIR}/defrcn_det_r101_base/model_final.pth                         \
    --save-dir ${SAVEDIR}/defrcn_det_r101_base
BASE_WEIGHT=${SAVEDIR}/defrcn_det_r101_base/model_reset_surgery.pth


# ------------------------------ Novel Fine-tuning ------------------------------- #
# --> 2. TFA-like, i.e. run seed0~9 for robust results (G-FSOD, 80 classes)
for seed in 0 1 2 3 4 5 6 7 8 9
do
    for shot in 1 2 3 5 10 30
    do
        python3 tools/create_config.py --dataset coco14 --config_root configs/coco     \
            --shot ${shot} --seed ${seed} --setting 'gfsod'
        CONFIG_PATH=configs/coco/defrcn_gfsod_r101_novel_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVEDIR}/defrcn_gfsod_r101_novel/tfa-like/${shot}shot_seed${seed}
        python3 main.py --num-gpus 8 --config-file ${CONFIG_PATH}                      \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}               \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}
        rm ${CONFIG_PATH}
        rm ${OUTPUT_DIR}/model_final.pth
    done
done
python3 tools/extract_results.py --res-dir ${SAVEDIR}/defrcn_gfsod_r101_novel/tfa-like --shot-list 1 2 3 5 10 30  # surmarize all results


# ------------------------------ Novel Fine-tuning ------------------------------- #  not necessary, just for the completeness of defrcn
# --> 3. TFA-like, i.e. run seed0~9 for robust results
BASE_WEIGHT=${SAVEDIR}/defrcn_det_r101_base/model_reset_remove.pth
for seed in 0 1 2 3 4 5 6 7 8 9
do
    for shot in 1 2 3 5 10 30
    do
        python3 tools/create_config.py --dataset coco14 --config_root configs/coco     \
            --shot ${shot} --seed ${seed} --setting 'fsod'
        CONFIG_PATH=configs/coco/defrcn_fsod_r101_novel_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVEDIR}/defrcn_fsod_r101_novel/tfa-like/${shot}shot_seed${seed}
        python3 main.py --num-gpus 8 --config-file ${CONFIG_PATH}                      \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}               \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}
        rm ${CONFIG_PATH}
        rm ${OUTPUT_DIR}/model_final.pth
    done
done
python3 tools/extract_results.py --res-dir ${SAVEDIR}/defrcn_fsod_r101_novel/tfa-like --shot-list 1 2 3 5 10 30  # surmarize all results
