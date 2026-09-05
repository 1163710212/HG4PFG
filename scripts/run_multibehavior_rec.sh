#!/usr/bin/env bash
set -euo pipefail

# Train two disjoint KuaiRec user-response simulators:
#   train: earlier 50% of every user's interactions
#   test:  later 50% of every user's interactions
# Per-user boundary records are balanced to make the global halves exact.

THIS_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
source "${THIS_SCRIPT_DIR}/_split_env.sh"

DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/dataset/KuaiRec}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/KuaiRec}"

TRAIN_FILE="${DATASET_DIR}/big_matrix_click.csv"
USER_META_FILE="${DATASET_DIR}/user_features.csv"
ITEM_META_FILE="${DATASET_DIR}/item_features.csv"

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
BATCH_SIZE="${BATCH_SIZE:-2048}"
N_WORKER="${N_WORKER:-6}"
POPULARITY_LOSS_COEF="${POPULARITY_LOSS_COEF:-0.1}"
POPULARITY_MARGIN="${POPULARITY_MARGIN:-1.5}"
POPULARITY_CATALOG_STEPS="${POPULARITY_CATALOG_STEPS:-300}"
POPULARITY_CATALOG_LR="${POPULARITY_CATALOG_LR:-0.001}"
POPULARITY_CATALOG_ANCHOR_COEF="${POPULARITY_CATALOG_ANCHOR_COEF:-0.5}"
FAILED_TASKS=()
COMPLETED_TASKS=()

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
                COMPLETION_PATH="${MODEL_PATH}.complete"
                INCOMPLETE_PATH="${MODEL_PATH}.incomplete"
                INIT_ARGS=()
                if [[ -n "${INIT_OUTPUT_DIR:-}" ]]; then
                    INIT_CHECKPOINT="${INIT_OUTPUT_DIR}/env/${ENVIRONMENT_SPLIT}/${FILE_KEY}.model.checkpoint"
                    if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
                        echo "Missing warm-start checkpoint: ${INIT_CHECKPOINT}" >&2
                        exit 1
                    fi
                    INIT_ARGS=(--init_checkpoint "${INIT_CHECKPOINT}")
                fi

                # A stale marker must never make a failed rerun look complete.
                rm -f -- "${COMPLETION_PATH}"
                printf 'environment_split=%s\nfile_key=%s\n' \
                    "${ENVIRONMENT_SPLIT}" "${FILE_KEY}" > "${INCOMPLETE_PATH}"

                echo "Training KuaiRec ${ENVIRONMENT_SPLIT}_env: ${FILE_KEY}"
                if PYTHONUNBUFFERED=1 "${PYTHON_BIN}" train_multibehavior.py \
                    --epoch "${EPOCHS}" \
                    --seed 619607 \
                    --lr "${LR}" \
                    --batch_size "${BATCH_SIZE}" \
                    --val_batch_size "${BATCH_SIZE}" \
                    --test_batch_size "${BATCH_SIZE}" \
                    --cuda "${CUDA_DEVICE}" \
                    --reader RecKRMBSeqReader \
                    --train_file "${TRAIN_FILE}" \
                    --user_meta_file "${USER_META_FILE}" \
                    --item_meta_file "${ITEM_META_FILE}" \
                    --dataset_dir "${DATASET_DIR}" \
                    --max_hist_seq_len 100 \
                    --data_separator ',' \
                    --meta_file_sep ',' \
                    --n_worker "${N_WORKER}" \
                    --environment_split "${ENVIRONMENT_SPLIT}" \
                    --train_environment_ratio 0.5 \
                    --environment_split_seed 619607 \
                    --val_holdout_per_user 5 \
                    --test_holdout_per_user 5 \
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
                    > "${LOG_PATH}" 2>&1
                then
                    if [[ ! -s "${MODEL_PATH}.checkpoint" ]]; then
                        echo "KuaiRec ${ENVIRONMENT_SPLIT}_env returned success but checkpoint is missing: ${MODEL_PATH}.checkpoint" >&2
                        FAILED_TASKS+=("${ENVIRONMENT_SPLIT}:${FILE_KEY}:missing-checkpoint")
                        continue
                    fi
                    if [[ ! -s "${MODEL_PATH}.metrics.json" ]]; then
                        echo "KuaiRec ${ENVIRONMENT_SPLIT}_env returned success but metrics are missing: ${MODEL_PATH}.metrics.json" >&2
                        FAILED_TASKS+=("${ENVIRONMENT_SPLIT}:${FILE_KEY}:missing-metrics")
                        continue
                    fi
                    if ! grep -Fq "environment split: ${ENVIRONMENT_SPLIT} " "${LOG_PATH}" || \
                       ! grep -Fq "final held-out test evaluation:" "${LOG_PATH}" || \
                       ! grep -Fq "Test result:" "${LOG_PATH}"
                    then
                        echo "KuaiRec ${ENVIRONMENT_SPLIT}_env returned success but completion evidence is missing: ${LOG_PATH}" >&2
                        FAILED_TASKS+=("${ENVIRONMENT_SPLIT}:${FILE_KEY}:incomplete-log")
                        continue
                    fi

                    printf 'environment_split=%s\nfile_key=%s\ncheckpoint=%s\n' \
                        "${ENVIRONMENT_SPLIT}" "${FILE_KEY}" "${MODEL_PATH}.checkpoint" \
                        > "${COMPLETION_PATH}"
                    rm -f -- "${INCOMPLETE_PATH}"
                    COMPLETED_TASKS+=("${ENVIRONMENT_SPLIT}:${FILE_KEY}")
                    echo "Finished KuaiRec ${ENVIRONMENT_SPLIT}_env; log: ${LOG_PATH}"
                else
                    TRAINING_RC=$?
                    FAILED_TASKS+=("${ENVIRONMENT_SPLIT}:${FILE_KEY}:exit=${TRAINING_RC}")
                    echo "KuaiRec ${ENVIRONMENT_SPLIT}_env failed with exit code ${TRAINING_RC}; continuing with the next environment. Log: ${LOG_PATH}" >&2
                fi
            done
        done
    done
done

if (( ${#FAILED_TASKS[@]} > 0 )); then
    echo "One or more KuaiRec environment tasks failed:" >&2
    printf '  %s\n' "${FAILED_TASKS[@]}" >&2
    echo "Completed tasks: ${#COMPLETED_TASKS[@]} / ${#ENVIRONMENT_SPLITS[@]}" >&2
    exit 1
fi

if (( ${#COMPLETED_TASKS[@]} != ${#ENVIRONMENT_SPLITS[@]} )); then
    echo "Expected ${#ENVIRONMENT_SPLITS[@]} completed KuaiRec environments, got ${#COMPLETED_TASKS[@]}" >&2
    exit 1
fi

echo "Successfully trained KuaiRec environments: ${COMPLETED_TASKS[*]}"
echo "KuaiRec train-environment checkpoints: ${OUTPUT_DIR}/env/train"
echo "KuaiRec test-environment checkpoints:  ${OUTPUT_DIR}/env/test"
