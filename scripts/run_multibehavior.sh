#!/usr/bin/env bash
set -euo pipefail

# Train two disjoint user-response simulators from the same chronological log:
#   train: earlier 50% of every user's interactions
#   test:  later 50% of every user's interactions
# The reader balances per-user boundary records so the global halves are exact.

THIS_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
source "${THIS_SCRIPT_DIR}/_split_env.sh"

KR_FLAG="Pure"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/dataset/Kuairand}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/Kuairand_${KR_FLAG}}"

TRAIN_FILE="${DATASET_DIR}/log_session_4_08_to_5_08_${KR_FLAG}.csv"
USER_META_FILE="${DATASET_DIR}/user_features_${KR_FLAG}_fillna.csv"
ITEM_META_FILE="${DATASET_DIR}/video_features_basic_${KR_FLAG}_fillna.csv"

for required_file in \
    "${TRAIN_FILE}" \
    "${USER_META_FILE}" \
    "${ITEM_META_FILE}" \
    "${DATASET_DIR}/user_pop_ratio.csv" \
    "${DATASET_DIR}/item_popularity.csv" \
    "${DATASET_DIR}/item_types.csv"
do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing required data file: ${required_file}" >&2
        exit 1
    fi
done

MODEL="KRMBUserResponse"
read -r -a ENVIRONMENT_SPLITS <<< "${ENVIRONMENT_SPLITS:-train test}"
LEARNING_RATES=(0.0001)
REGULARIZATIONS=(0)
TRANSFORMER_LAYERS=(2)
EPOCHS="${EPOCHS:-10}"
POPULARITY_LOSS_COEF="${POPULARITY_LOSS_COEF:-0.1}"
POPULARITY_MARGIN="${POPULARITY_MARGIN:-1.5}"
POPULARITY_CATALOG_STEPS="${POPULARITY_CATALOG_STEPS:-300}"
POPULARITY_CATALOG_LR="${POPULARITY_CATALOG_LR:-0.001}"
POPULARITY_CATALOG_ANCHOR_COEF="${POPULARITY_CATALOG_ANCHOR_COEF:-0.5}"

for ENVIRONMENT_SPLIT in "${ENVIRONMENT_SPLITS[@]}"
do
    ENV_OUTPUT_DIR="${OUTPUT_DIR}/env/${ENVIRONMENT_SPLIT}"
    LOG_DIR="${ENV_OUTPUT_DIR}/log"
    mkdir -p "${LOG_DIR}"

    for LR in "${LEARNING_RATES[@]}"
    do
        for REG in "${REGULARIZATIONS[@]}"
        do
            for N_LAYER in "${TRANSFORMER_LAYERS[@]}"
            do
                FILE_KEY="user_${MODEL}_lr${LR}_reg${REG}_nlayer${N_LAYER}"
                MODEL_PATH="${ENV_OUTPUT_DIR}/${FILE_KEY}.model"
                LOG_PATH="${LOG_DIR}/${FILE_KEY}.model.log"
                INIT_ARGS=()
                if [[ -n "${INIT_OUTPUT_DIR:-}" ]]; then
                    INIT_CHECKPOINT="${INIT_OUTPUT_DIR}/env/${ENVIRONMENT_SPLIT}/${FILE_KEY}.model.checkpoint"
                    if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
                        echo "Missing warm-start checkpoint: ${INIT_CHECKPOINT}" >&2
                        exit 1
                    fi
                    INIT_ARGS=(--init_checkpoint "${INIT_CHECKPOINT}")
                fi

                echo "Training ${ENVIRONMENT_SPLIT}_env: ${FILE_KEY}"
                "${PYTHON_BIN}" train_multibehavior.py \
                    --epoch "${EPOCHS}" \
                    --seed 619607 \
                    --lr "${LR}" \
                    --batch_size 512 \
                    --val_batch_size 512 \
                    --test_batch_size 512 \
                    --cuda "${CUDA_DEVICE}" \
                    --reader KRMBSeqReader \
                    --train_file "${TRAIN_FILE}" \
                    --user_meta_file "${USER_META_FILE}" \
                    --item_meta_file "${ITEM_META_FILE}" \
                    --dataset_dir "${DATASET_DIR}" \
                    --max_hist_seq_len 100 \
                    --data_separator ',' \
                    --meta_file_sep ',' \
                    --n_worker "${N_WORKER:-0}" \
                    --environment_split "${ENVIRONMENT_SPLIT}" \
                    --train_environment_ratio 0.5 \
                    --environment_split_seed 619607 \
                    --val_holdout_per_user 2 \
                    --test_holdout_per_user 2 \
                    --model "${MODEL}" \
                    --loss bce \
                    --l2_coef "${REG}" \
                    --model_path "${MODEL_PATH}" \
                    "${INIT_ARGS[@]}" \
                    --save_with_val \
                    --early_stop_patience 3 \
                    --user_latent_dim 32 \
                    --item_latent_dim 32 \
                    --enc_dim 64 \
                    --attn_n_head 4 \
                    --transformer_d_forward 64 \
                    --transformer_n_layer "${N_LAYER}" \
                    --state_hidden_dims 128 \
                    --scorer_hidden_dims 128 32 \
                    --dropout_rate 0.1 \
                    --popularity_loss_coef "${POPULARITY_LOSS_COEF}" \
                    --popularity_margin "${POPULARITY_MARGIN}" \
                    --popularity_catalog_steps "${POPULARITY_CATALOG_STEPS}" \
                    --popularity_catalog_lr "${POPULARITY_CATALOG_LR}" \
                    --popularity_catalog_anchor_coef "${POPULARITY_CATALOG_ANCHOR_COEF}" \
                    > "${LOG_PATH}"

                echo "Finished ${ENVIRONMENT_SPLIT}_env; log: ${LOG_PATH}"
            done
        done
    done
done

echo "Train-environment checkpoints: ${OUTPUT_DIR}/env/train"
echo "Test-environment checkpoints:  ${OUTPUT_DIR}/env/test"
