#!/usr/bin/env bash
set -euo pipefail

THIS_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${THIS_SCRIPT_DIR}/_split_env.sh"

DATASET="Kuairand_Pure"
setup_split_environment "Kuairand" "Kuairand_Pure" \
    "user_KRMBUserResponse_lr0.0001_reg0_nlayer2" "run_multibehavior.sh"

ENV_CLASS="KREnvironment_WholeSession_GPU"
POLICY_CLASS="HRL4PFGPolicy"
CRITIC_CLASS="HRL4PFGCritic"
BUFFER_CLASS="HRL4PFGBuffer"
AGENT_CLASS="HRL4PFGPPO"

# Preserve train_hppo_discrete.sh's simulator, state tracker and rollout
# settings. M, lambda_f and lambda_g control the macro interval,
# entropy-normalized fairness reward and low-level guidance reward.
MAX_STEP="${MAX_STEP:-30}"
SLATE_SIZE="${SLATE_SIZE:-10}"
EPISODE_BATCH_SIZE=64
ITEM_CORRELATION="${ITEM_CORRELATION:-0.2}"
AD_TEMPER_PENALTY="${AD_TEMPER_PENALTY:-1.0}"
BUFFER_SIZE="${BUFFER_SIZE:-5000}"
GAMMA="${GAMMA:-0.9}"
N_ITER=30000
TRAIN_EVERY_N_STEP="${TRAIN_EVERY_N_STEP:-20}"
EXPLORE_EPSILON="${EXPLORE_EPSILON:-0.01}"
EXPLORE_RATE="${EXPLORE_RATE:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ACTION_STD="${ACTION_STD:-0.1}"
GOAL_STD_MIN="${GOAL_STD_MIN:-0.02}"
GOAL_STD_MAX="${GOAL_STD_MAX:-0.2}"
AD_BOUND="${AD_BOUND:-0.3}"
ACTOR_LR="${ACTOR_LR:-0.00008}"
CRITIC_LR="${CRITIC_LR:-0.001}"
REGULARIZATION="${REGULARIZATION:-0.00001}"
TARGET_MITIGATE_COEF="${TARGET_MITIGATE_COEF:-0.02}"
TRAIN_EPOCH_NUM="${TRAIN_EPOCH_NUM:-4}"
EPS_CLIP="${EPS_CLIP:-0.8}"
HIGH_ENTROPY_COEF="${HIGH_ENTROPY_COEF:-0.0005}"
LOW_ENTROPY_COEF="${LOW_ENTROPY_COEF:-0.0005}"
SEED="${SEED:-12}"
LOW_LEVEL_ONLY="${LOW_LEVEL_ONLY:-0}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-1}"
DRY_RUN="${DRY_RUN:-0}"

CUDA_DEVICES=(0 1)
LAMBDA_FAIRNESS_VALUES=(0.1)
#(0.04 0.06)
LAMBDA_GUIDE_VALUES=(0.001 0.04)
#(0 0.02 0.04 0.06 0.08 0.1)
SUBGOAL_INTERVAL_VALUES=(4)
#(1 2 3 4 5)

AGENT_MODE_ARGS=()
MODE_SUFFIX=""
case "${LOW_LEVEL_ONLY}" in
    0) ;;
    1)
        AGENT_MODE_ARGS+=(--low_level_only)
        MODE_SUFFIX="_low_only"
        ;;
    *)
        echo "LOW_LEVEL_ONLY must be 0 or 1" >&2
        exit 1
        ;;
esac

if [[ ! "${SLOTS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SLOTS_PER_GPU must be a positive integer" >&2
    exit 1
fi
case "${DRY_RUN}" in
    0|1) ;;
    *)
        echo "DRY_RUN must be 0 or 1" >&2
        exit 1
        ;;
esac
if (( BUFFER_SIZE < EPISODE_BATCH_SIZE * TRAIN_EVERY_N_STEP )); then
    echo "BUFFER_SIZE must be at least EPISODE_BATCH_SIZE * TRAIN_EVERY_N_STEP" >&2
    exit 1
fi

EXPECTED_TASKS=$((${#LAMBDA_FAIRNESS_VALUES[@]} * ${#LAMBDA_GUIDE_VALUES[@]} * ${#SUBGOAL_INTERVAL_VALUES[@]}))
TOTAL_SLOTS=$((${#CUDA_DEVICES[@]} * SLOTS_PER_GPU))
SWEEP_TAG="${SWEEP_TAG:-$(date '+%Y%m%d_%H%M%S')}"
SWEEP_DIR="${SWEEP_DIR:-${OUTPUT_DIR}/sweeps/hrl4pfg_${SWEEP_TAG}}"
RUN_ROOT="${RUN_ROOT:-${OUTPUT_DIR}/agents/hrl4pfg_sweep_${SWEEP_TAG}}"
TASKS_FILE="${SWEEP_DIR}/tasks.tsv"
FAILED_FILE="${SWEEP_DIR}/failed.tsv"
DISPATCH_LOG="${SWEEP_DIR}/dispatch.log"

mkdir -p "${SWEEP_DIR}"
printf 'task_id\tworker_slot\tgpu\tgpu_slot\tlambda_fairness\tlambda_guide\tsubgoal_interval\tagent_dir\n' > "${TASKS_FILE}"
printf 'task_id\tgpu\tlambda_fairness\tlambda_guide\tsubgoal_interval\texit_code\n' > "${FAILED_FILE}"
: > "${DISPATCH_LOG}"

task_id=0
for lambda_fairness in "${LAMBDA_FAIRNESS_VALUES[@]}"
do
    for lambda_guide in "${LAMBDA_GUIDE_VALUES[@]}"
    do
        for subgoal_interval in "${SUBGOAL_INTERVAL_VALUES[@]}"
        do
            worker_slot=$((task_id % TOTAL_SLOTS))
            gpu_index=$((worker_slot / SLOTS_PER_GPU))
            gpu_slot=$((worker_slot % SLOTS_PER_GPU))
            gpu="${CUDA_DEVICES[${gpu_index}]}"
            file_key="${AGENT_CLASS}${MODE_SUFFIX}_M${subgoal_interval}_lf${lambda_fairness}_lg${lambda_guide}_actor_lr_${ACTOR_LR}_seed${SEED}_ms${MAX_STEP}_ni${N_ITER}"
            agent_dir="${RUN_ROOT}/${file_key}"
            printf '%d\t%d\t%s\t%d\t%s\t%s\t%s\t%s\n' \
                "${task_id}" "${worker_slot}" "${gpu}" "${gpu_slot}" \
                "${lambda_fairness}" "${lambda_guide}" "${subgoal_interval}" \
                "${agent_dir}" >> "${TASKS_FILE}"
            task_id=$((task_id + 1))
        done
    done
done

if (( task_id != EXPECTED_TASKS )); then
    echo "Expected ${EXPECTED_TASKS} tasks, generated ${task_id}" >&2
    exit 1
fi

run_one() {
    local current_task_id="$1"
    local gpu="$2"
    local lambda_fairness="$3"
    local lambda_guide="$4"
    local subgoal_interval="$5"
    local agent_dir="$6"

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY_RUN task=%s gpu=%s lambda_fairness=%s lambda_guide=%s subgoal_interval=%s\n' \
            "${current_task_id}" "${gpu}" "${lambda_fairness}" \
            "${lambda_guide}" "${subgoal_interval}"
        return 0
    fi

    local normalized_env_dir="${agent_dir}/environment_logs"
    local train_env_log="${normalized_env_dir}/train.model.log"
    local test_env_log="${normalized_env_dir}/test.model.log"
    mkdir -p "${normalized_env_dir}"
    normalize_environment_log "${TRAIN_ENV_LOG}" "${train_env_log}"
    normalize_environment_log "${TEST_ENV_LOG}" "${test_env_log}"

    "${PYTHON_BIN}" -u train_actor_critic.py \
        --env_class "${ENV_CLASS}" \
        --policy_class "${POLICY_CLASS}" \
        --critic_class "${CRITIC_CLASS}" \
        --buffer_class "${BUFFER_CLASS}" \
        --agent_class "${AGENT_CLASS}" \
        --dataset "${DATASET}" \
        --dataset_dir "${DATASET_DIR}" \
        --uirm_log_path "${train_env_log}" \
        --test_uirm_log_path "${test_env_log}" \
        --aug_weight 0 \
        --seed "${SEED}" \
        --ad_bound "${AD_BOUND}" \
        --ad_temper_penalty "${AD_TEMPER_PENALTY}" \
        --cuda "${gpu}" \
        --max_step_per_episode "${MAX_STEP}" \
        --initial_temper "${MAX_STEP}" \
        --slate_size "${SLATE_SIZE}" \
        --episode_batch_size "${EPISODE_BATCH_SIZE}" \
        --item_correlation "${ITEM_CORRELATION}" \
        --single_response \
        --policy_action_hidden 256 64 \
        --action_std_init "${ACTION_STD}" \
        --goal_std_min "${GOAL_STD_MIN}" \
        --goal_std_max "${GOAL_STD_MAX}" \
        --state_user_latent_dim 16 \
        --state_item_latent_dim 16 \
        --state_transformer_enc_dim 32 \
        --state_transformer_n_head 4 \
        --state_transformer_d_forward 64 \
        --state_transformer_n_layer 3 \
        --state_dropout_rate 0 \
        --critic_hidden_dims 256 64 \
        --critic_dropout_rate 0.1 \
        --buffer_size "${BUFFER_SIZE}" \
        --gamma "${GAMMA}" \
        --reward_func get_immediate_reward \
        --n_iter "${N_ITER}" \
        --train_every_n_step "${TRAIN_EVERY_N_STEP}" \
        --initial_epsilon "${EXPLORE_EPSILON}" \
        --final_epsilon "${EXPLORE_EPSILON}" \
        --elbow_epsilon 0.1 \
        --explore_rate "${EXPLORE_RATE}" \
        --check_episode 10 \
        --save_episode 200 \
        --test_repeat "${TEST_REPEAT}" \
        --test_num_users "${TEST_NUM_USERS}" \
        --test_seed "${TEST_SEED}" \
        --save_path "${agent_dir}/model" \
        --actor_lr "${ACTOR_LR}" \
        --actor_decay "${REGULARIZATION}" \
        --batch_size "${BATCH_SIZE}" \
        --critic_lr "${CRITIC_LR}" \
        --critic_decay "${REGULARIZATION}" \
        --target_mitigate_coef "${TARGET_MITIGATE_COEF}" \
        --train_epoch_num "${TRAIN_EPOCH_NUM}" \
        --eps_clip "${EPS_CLIP}" \
        --high_entropy_coef "${HIGH_ENTROPY_COEF}" \
        --low_entropy_coef "${LOW_ENTROPY_COEF}" \
        --subgoal_interval "${subgoal_interval}" \
        --lambda_fairness "${lambda_fairness}" \
        --lambda_guide "${lambda_guide}" \
        "${AGENT_MODE_ARGS[@]}" \
        > "${agent_dir}/log" 2>&1
}

worker() {
    local assigned_slot="$1"
    local current_task_id current_slot gpu gpu_slot
    local lambda_fairness lambda_guide subgoal_interval agent_dir
    while IFS=$'\t' read -r current_task_id current_slot gpu gpu_slot \
        lambda_fairness lambda_guide subgoal_interval agent_dir
    do
        [[ "${current_task_id}" == "task_id" ]] && continue
        [[ "${current_slot}" != "${assigned_slot}" ]] && continue

        printf '%s START task=%s gpu=%s gpu_slot=%s lf=%s lg=%s M=%s\n' \
            "$(date '+%F %T')" "${current_task_id}" "${gpu}" "${gpu_slot}" \
            "${lambda_fairness}" "${lambda_guide}" "${subgoal_interval}" \
            >> "${DISPATCH_LOG}"
        if run_one "${current_task_id}" "${gpu}" "${lambda_fairness}" \
            "${lambda_guide}" "${subgoal_interval}" "${agent_dir}"
        then
            printf '%s DONE task=%s gpu=%s lf=%s lg=%s M=%s\n' \
                "$(date '+%F %T')" "${current_task_id}" "${gpu}" \
                "${lambda_fairness}" "${lambda_guide}" "${subgoal_interval}" \
                >> "${DISPATCH_LOG}"
        else
            local exit_code=$?
            printf '%s FAIL task=%s gpu=%s lf=%s lg=%s M=%s exit=%s\n' \
                "$(date '+%F %T')" "${current_task_id}" "${gpu}" \
                "${lambda_fairness}" "${lambda_guide}" "${subgoal_interval}" \
                "${exit_code}" >> "${DISPATCH_LOG}"
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${current_task_id}" "${gpu}" "${lambda_fairness}" \
                "${lambda_guide}" "${subgoal_interval}" "${exit_code}" \
                >> "${FAILED_FILE}"
        fi
    done < "${TASKS_FILE}"
}

printf 'HRL4PFG sweep: tasks=%d, GPUs=%s, slots_per_gpu=%d, total_workers=%d, dry_run=%s\n' \
    "${EXPECTED_TASKS}" "${CUDA_DEVICES[*]}" "${SLOTS_PER_GPU}" \
    "${TOTAL_SLOTS}" "${DRY_RUN}"
printf 'Sweep audit directory: %s\n' "${SWEEP_DIR}"
printf 'Sweep model directory: %s\n' "${RUN_ROOT}"

worker_pids=()
for ((worker_slot = 0; worker_slot < TOTAL_SLOTS; worker_slot++))
do
    worker "${worker_slot}" &
    worker_pids+=("$!")
done

worker_error=0
for worker_pid in "${worker_pids[@]}"
do
    if ! wait "${worker_pid}"
    then
        worker_error=1
    fi
done

failed_count="$(awk 'NR > 1 {count++} END {print count + 0}' "${FAILED_FILE}")"
printf 'HRL4PFG sweep finished: tasks=%d, failed=%s, audit_dir=%s\n' \
    "${EXPECTED_TASKS}" "${failed_count}" "${SWEEP_DIR}"
if (( worker_error != 0 || failed_count != 0 )); then
    exit 1
fi
