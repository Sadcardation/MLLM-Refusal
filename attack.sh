#!/bin/bash

# case 1: shadow question is exactly user question from dataset, user question is image related (same with shadow question)
# case 2: shadow question is exactly user question from dataset, user question is image irrelevant (sampled from commonsense_qa)
# case 3: shadow questions are similar user questions, user question is image related question from dataset
# case 4: shadow questions are similar user questions, user question is image irrelevant (sampled from commonsense_qa)
# case 5: shadow questions are general user questions, user question is image related question from dataset
# case 6: shadow questions are general user questions, user question is image irrelevant (sampled from commonsense_qa)

# Case 1
EPS=8
N_ITERS=1000
LR=0.007
LR_STR=$(echo $LR | sed 's/\./p/')
DATASET_NAME='VQA'
ATTACK_MODE='normal'
OPT=FGSM
SAMPLE=50 # of shadow questions, useless in case 1 and 2
QBATCH=1
MODEL='LLaVA'
CASE=1

CUDA_VISIBLE_DEVICES=0 python scripts/mllm_refusal.py \
    --model llava \
    --database_name $DATASET_NAME \
    --file_path ./datasets/VQAv2/sampled_data_100.xlsx \
    --images_path ./datasets/VQAv2/Images/mscoco/val2014 \
    --log_dir REFUSAL_know_q_${ATTACK_MODE}_lr_${LR_STR}_${N_ITERS}_iters_eps_${EPS}_${OPT}_${DATASET_NAME}_${MODEL}_case_${CASE} \
    --eps $EPS \
    --attack_mode $ATTACK_MODE \
    --n_iters $N_ITERS \
    --optimizer $OPT \
    --alpha $LR \
    --q_batch $QBATCH \
    --case $CASE
    # --checkpoint


# Case 3
EPS=8
N_ITERS=1500
LR=0.005
LR_STR=$(echo $LR | sed 's/\./p/')
DATASET_NAME='VQA'
ATTACK_MODE='normal_mean'
OPT=FGSM
SAMPLE=10
QBATCH=3
MODEL='LLaVA'
CASE=3

# Run the python script with the macro variables
CUDA_VISIBLE_DEVICES=0 python scripts/mllm_refusal.py \
    --model llava \
    --database_name $DATASET_NAME \
    --file_path ./datasets/VQAv2/sampled_data_100.xlsx \
    --images_path ./datasets/VQAv2/Images/mscoco/val2014 \
    --log_dir REFUSAL_unknow_q_${SAMPLE}_${QBATCH}_${ATTACK_MODE}_lr_${LR_STR}_${N_ITERS}_iters_eps_${EPS}_${OPT}_${DATASET_NAME}_${MODEL}_case_${CASE} \
    --eps $EPS \
    --attack_mode $ATTACK_MODE \
    --n_iters $N_ITERS \
    --optimizer $OPT \
    --alpha $LR \
    --q_batch $QBATCH \
    --case $CASE
    # --checkpoint

# Case 5
EPS=8
N_ITERS=1500
LR=0.005
LR_STR=$(echo $LR | sed 's/\./p/')
DATASET_NAME='VQA'
ATTACK_MODE='normal_mean'
OPT=FGSM
QUOTA=10
SAMPLE=50
QBATCH=3
MODEL='LLaVA'
CASE=5

# Run the python script with the macro variables
CUDA_VISIBLE_DEVICES=7 python scripts/mllm_refusal.py \
    --model llava \
    --database_name $DATASET_NAME \
    --file_path ./datasets/VQAv2/sampled_data_100.xlsx \
    --images_path ./datasets/VQAv2/Images/mscoco/val2014 \
    --log_dir REFUSAL_unknow_q_${SAMPLE}_${QBATCH}_${ATTACK_MODE}_lr_${LR_STR}_${N_ITERS}_iters_eps_${EPS}_${OPT}_${DATASET_NAME}_${MODEL}_case_${CASE} \
    --eps $EPS \
    --attack_mode $ATTACK_MODE \
    --n_iters $N_ITERS \
    --optimizer $OPT \
    --alpha $LR \
    --q_batch $QBATCH \
    --case $CASE
    # --checkpoint