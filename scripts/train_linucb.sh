#!/usr/bin/env bash
set -euo pipefail

THIS_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${THIS_SCRIPT_DIR}/_split_env.sh"

DATASET="Kuairand"
setup_split_environment "Kuairand" "Kuairand_Pure" \
    "user_KRMBUserResponse_lr0.0001_reg0_nlayer2" "run_multibehavior.sh"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

# Keep the simulator contract identical to train_a2c.sh.
ENV_CLASS='KREnvironment_WholeSession_GPU'
MAX_STEP="${MAX_STEP:-30}"
SLATE_SIZE="${SLATE_SIZE:-10}"
EPISODE_BATCH_SIZE="${EPISODE_BATCH_SIZE:-64}"
RHO="${RHO:-0.2}"
AD_TEMPER_PENALTY="${AD_TEMPER_PENALTY:-1.0}"

# Algorithm 1 in LinUCB.pdf.  Six context dimensions mirror the paper's
# five reduced user-preference features plus a constant feature.
LINUCB_ALPHA="${LINUCB_ALPHA:-1.0}"
LINUCB_RIDGE="${LINUCB_RIDGE:-1.0}"
LINUCB_CONTEXT_DIM="${LINUCB_CONTEXT_DIM:-6}"
LINUCB_SCORE_CHUNK_SIZE="${LINUCB_SCORE_CHUNK_SIZE:-4096}"
N_ITER="${N_ITER:-30000}"
SEED="${SEED:-12}"

file_key='LinUCB'
mkdir -p "${output_path}agents/${file_key}/"

"${PYTHON_BIN}" train_linucb.py \
    --env_class "${ENV_CLASS}" \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --cuda "${CUDA_DEVICE}" \
    --max_step_per_episode "${MAX_STEP}" \
    --initial_temper "${MAX_STEP}" \
    --ad_temper_penalty "${AD_TEMPER_PENALTY}" \
    --uirm_log_path "${TRAIN_ENV_LOG}" \
    --test_uirm_log_path "${TEST_ENV_LOG}" \
    --dataset_dir "${DATASET_DIR}" \
    --test_n_step "${TEST_N_STEP}" \
    --test_repeat "${TEST_REPEAT}" \
    --test_num_users "${TEST_NUM_USERS}" \
    --test_seed "${TEST_SEED}" \
    --slate_size "${SLATE_SIZE}" \
    --episode_batch_size "${EPISODE_BATCH_SIZE}" \
    --item_correlation "${RHO}" \
    --single_response \
    --reward_func get_immediate_reward \
    --n_iter "${N_ITER}" \
    --check_episode 10 \
    --save_episode 200 \
    --save_path "${output_path}agents/${file_key}/model" \
    --linucb_alpha "${LINUCB_ALPHA}" \
    --linucb_ridge "${LINUCB_RIDGE}" \
    --linucb_context_dim "${LINUCB_CONTEXT_DIM}" \
    --linucb_score_chunk_size "${LINUCB_SCORE_CHUNK_SIZE}" \
    > "${output_path}agents/${file_key}/log"
