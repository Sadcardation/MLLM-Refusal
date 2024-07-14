CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
    --results_path models/LLaVA \
    --result_index REFUSAL_unknow_q_50_3_normal_mean_lr_0p005_2_iters_eps_8_FGSM_VQA_Llava_case_5 \
    --candidates 3 \
    --case 5 \
    --model llava \
    --dataset VQA \
    --need_query