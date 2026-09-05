#!/usr/bin/env bash
set -euo pipefail

THIS_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${THIS_SCRIPT_DIR}/_split_env.sh"

# HRL4PFG sweep over both simulator environments and AD temper penalties.
# The current HRL4PFG lambda/subgoal settings from train_hrl4pfg.sh are kept
# literal here and can be expanded by editing the arrays below.
CUDA_DEVICES=(0 1)
SLOTS_PER_GPU="${SLOTS_PER_GPU:-1}"
DRY_RUN="${DRY_RUN:-0}"
ENVIRONMENTS=(kuairand kuairec)
#(kuairand kuairec)
AD_TEMPER_PENALTIES=(1)
#(1 2 3 4 5 6)
LAMBDA_FAIRNESS_VALUES=(0.1)
LAMBDA_GUIDE_VALUES=(0.04)
SUBGOAL_INTERVAL_VALUES=(5)

MAX_STEP="${MAX_STEP:-10}"
# 30 50 10
SLATE_SIZE=10
EPISODE_BATCH_SIZE=64
ITEM_CORRELATION=0.2
BUFFER_SIZE=5000
GAMMA=0.9
N_ITER=30000
TRAIN_EVERY_N_STEP=20
EXPLORE_EPSILON=0.01
EXPLORE_RATE=1.0
BATCH_SIZE=128
ACTION_STD=0.1
GOAL_STD_MIN=0.02
GOAL_STD_MAX=0.2
AD_BOUND=0.3
ACTOR_LR=0.00008
CRITIC_LR=0.001
REGULARIZATION=0.00001
TARGET_MITIGATE_COEF=0.02
TRAIN_EPOCH_NUM=4
EPS_CLIP=0.8
HIGH_ENTROPY_COEF=0.0005
LOW_ENTROPY_COEF=0.0005
SEED="${SEED:-13}"
LOW_LEVEL_ONLY="${LOW_LEVEL_ONLY:-0}"
# The historical v5 mapping remains the default.  Set
# TARGET_PROJECTION_MODE=soft_popularity to opt into the v4-style gradual,
# preference-compatible mapping without changing any other training setting.
TARGET_PROJECTION_MODE="${TARGET_PROJECTION_MODE:-catalog_nearest}"
GOAL_RESIDUAL_SCALE="${GOAL_RESIDUAL_SCALE:-2.0}"
TARGET_CANDIDATE_SIZE="${TARGET_CANDIDATE_SIZE:-128}"
SOFT_TARGET_TAIL_ADVANCE="${SOFT_TARGET_TAIL_ADVANCE:-0.2}"
SOFT_TARGET_POPULARITY_COEF="${SOFT_TARGET_POPULARITY_COEF:-4.0}"

if [[ ! "${MAX_STEP}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_STEP must be a positive integer" >&2
    exit 1
fi
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "SEED must be a non-negative integer" >&2
    exit 1
fi
if [[ ! "${SLOTS_PER_GPU}" =~ ^[12]$ ]]; then
    echo "SLOTS_PER_GPU must be 1 or 2; the hard per-GPU cap is 2." >&2
    exit 1
fi
case "${DRY_RUN}" in
    0|1) ;;
    *) echo "DRY_RUN must be 0 or 1" >&2; exit 1 ;;
esac
case "${LOW_LEVEL_ONLY}" in
    0)
        AGENT_MODE_ARGS=()
        MODE_SUFFIX=""
        ;;
    1)
        AGENT_MODE_ARGS=(--low_level_only)
        MODE_SUFFIX="_low_only"
        ;;
    *)
        echo "LOW_LEVEL_ONLY must be 0 or 1" >&2
        exit 1
        ;;
esac
case "${TARGET_PROJECTION_MODE}" in
    catalog_nearest)
        TARGET_MAPPING_ARGS=()
        TARGET_MAPPING_SUFFIX=""
        ;;
    hard_tail)
        TARGET_MAPPING_ARGS=()
        TARGET_MAPPING_SUFFIX="_hard_tail"
        ;;
    soft_popularity)
        TARGET_MAPPING_ARGS=(--align_item_metadata)
        TARGET_MAPPING_SUFFIX="_v4soft"
        ;;
    *)
        echo "TARGET_PROJECTION_MODE must be catalog_nearest, hard_tail, or soft_popularity" >&2
        exit 1
        ;;
esac
if [[ ! "${TARGET_CANDIDATE_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TARGET_CANDIDATE_SIZE must be a positive integer" >&2
    exit 1
fi
if (( BUFFER_SIZE < EPISODE_BATCH_SIZE * TRAIN_EVERY_N_STEP )); then
    echo "BUFFER_SIZE must be at least EPISODE_BATCH_SIZE * TRAIN_EVERY_N_STEP" >&2
    exit 1
fi

TOTAL_SLOTS=$((${#CUDA_DEVICES[@]} * SLOTS_PER_GPU))
EXPECTED_TASKS=$((${#ENVIRONMENTS[@]} * ${#AD_TEMPER_PENALTIES[@]} * ${#LAMBDA_FAIRNESS_VALUES[@]} * ${#LAMBDA_GUIDE_VALUES[@]} * ${#SUBGOAL_INTERVAL_VALUES[@]}))
SWEEP_TAG="${SWEEP_TAG:-$(date '+%Y%m%d_%H%M%S')}"
SWEEP_DIR="${SWEEP_DIR:-${PROJECT_ROOT}/output/sweeps/hrl4pfg_adp_${SWEEP_TAG}}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/output/hrl4pfg_adp_sweep_${SWEEP_TAG}}"
TASKS_FILE="${SWEEP_DIR}/tasks.tsv"
FAILED_FILE="${SWEEP_DIR}/failed.tsv"
DISPATCH_LOG="${SWEEP_DIR}/dispatch.log"

mkdir -p "${SWEEP_DIR}" "${RUN_ROOT}"
printf 'task_id\tworker_slot\tgpu\tgpu_slot\tenvironment\tad_temper_penalty\tlambda_fairness\tlambda_guide\tsubgoal_interval\tagent_dir\n' > "${TASKS_FILE}"
printf 'task_id\tgpu\tenvironment\tad_temper_penalty\tlambda_fairness\tlambda_guide\tsubgoal_interval\texit_code\n' > "${FAILED_FILE}"
: > "${DISPATCH_LOG}"

task_id=0
for environment in "${ENVIRONMENTS[@]}"; do
    for penalty in "${AD_TEMPER_PENALTIES[@]}"; do
        for lambda_fairness in "${LAMBDA_FAIRNESS_VALUES[@]}"; do
            for lambda_guide in "${LAMBDA_GUIDE_VALUES[@]}"; do
                for subgoal_interval in "${SUBGOAL_INTERVAL_VALUES[@]}"; do
                    worker_slot=$((task_id % TOTAL_SLOTS))
                    gpu_index=$((worker_slot / SLOTS_PER_GPU))
                    gpu_slot=$((worker_slot % SLOTS_PER_GPU))
                    gpu="${CUDA_DEVICES[${gpu_index}]}"
                    file_key="HRL4PFGPPO${MODE_SUFFIX}${TARGET_MAPPING_SUFFIX}_${environment}_adp${penalty}_M${subgoal_interval}_lf${lambda_fairness}_lg${lambda_guide}_seed${SEED}_ni${N_ITER}"
                    agent_dir="${RUN_ROOT}/${environment}/adp_${penalty}/${file_key}"
                    printf '%d\t%d\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                        "${task_id}" "${worker_slot}" "${gpu}" "${gpu_slot}" \
                        "${environment}" "${penalty}" "${lambda_fairness}" \
                        "${lambda_guide}" "${subgoal_interval}" "${agent_dir}" \
                        >> "${TASKS_FILE}"
                    task_id=$((task_id + 1))
                done
            done
        done
    done
done

if (( task_id != EXPECTED_TASKS )); then
    echo "Expected ${EXPECTED_TASKS} tasks, generated ${task_id}" >&2
    exit 1
fi

environment_config() {
    local environment="$1"
    case "${environment}" in
        kuairand)
            ENV_DATASET="Kuairand_Pure"
            ENV_DATASET_DIR="${PROJECT_ROOT}/dataset/Kuairand"
            ENV_CANONICAL_OUTPUT="${PROJECT_ROOT}/output/Kuairand_Pure"
            ;;
        kuairec)
            ENV_DATASET="KuaiRec"
            ENV_DATASET_DIR="${PROJECT_ROOT}/dataset/KuaiRec"
            ENV_CANONICAL_OUTPUT="${PROJECT_ROOT}/output/KuaiRec"
            ;;
        *)
            echo "Unknown environment: ${environment}" >&2
            return 2
            ;;
    esac
}

validate_environment_artifacts() {
    local canonical_output="$1"
    local model_key="user_KRMBUserResponse_lr0.0001_reg0_nlayer2"
    local split
    for split in train test; do
        if [[ ! -f "${canonical_output}/env/${split}/log/${model_key}.model.log" || \
              ! -f "${canonical_output}/env/${split}/${model_key}.model.checkpoint" ]]; then
            echo "Missing ${split} environment artifacts under ${canonical_output}/env" >&2
            return 3
        fi
    done
    if [[ -e "${canonical_output}/env/train/${model_key}.model.incomplete" || \
          -e "${canonical_output}/env/test/${model_key}.model.incomplete" ]]; then
        echo "Environment training is incomplete under ${canonical_output}/env" >&2
        return 4
    fi
    if [[ -e "${canonical_output}/env/train/${model_key}.model.complete" || \
          -e "${canonical_output}/env/test/${model_key}.model.complete" ]]; then
        for split in train test; do
            if [[ ! -f "${canonical_output}/env/${split}/${model_key}.model.complete" ]]; then
                echo "Missing ${split} completion marker under ${canonical_output}/env" >&2
                return 5
            fi
        done
    fi
}

run_one() {
    local current_task_id="$1"
    local gpu="$2"
    local environment="$3"
    local penalty="$4"
    local lambda_fairness="$5"
    local lambda_guide="$6"
    local subgoal_interval="$7"
    local agent_dir="$8"

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY_RUN task=%s gpu=%s environment=%s ad_temper_penalty=%s lf=%s lg=%s M=%s max_step=%s seed=%s\n' \
            "${current_task_id}" "${gpu}" "${environment}" "${penalty}" \
            "${lambda_fairness}" "${lambda_guide}" "${subgoal_interval}" \
            "${MAX_STEP}" "${SEED}"
        return 0
    fi

    environment_config "${environment}"
    validate_environment_artifacts "${ENV_CANONICAL_OUTPUT}"
    normalized_env_dir="${agent_dir}/environment_logs"
    train_env_log="${normalized_env_dir}/train.model.log"
    test_env_log="${normalized_env_dir}/test.model.log"
    mkdir -p "${normalized_env_dir}"
    normalize_environment_log \
        "${ENV_CANONICAL_OUTPUT}/env/train/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.log" \
        "${train_env_log}"
    normalize_environment_log \
        "${ENV_CANONICAL_OUTPUT}/env/test/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.log" \
        "${test_env_log}"

    "${PYTHON_BIN}" -u train_actor_critic.py \
        --env_class KREnvironment_WholeSession_GPU \
        --policy_class HRL4PFGPolicy \
        --critic_class HRL4PFGCritic \
        --buffer_class HRL4PFGBuffer \
        --agent_class HRL4PFGPPO \
        --dataset "${ENV_DATASET}" \
        --dataset_dir "${ENV_DATASET_DIR}" \
        --uirm_log_path "${train_env_log}" \
        --test_uirm_log_path "${test_env_log}" \
        --aug_weight 0 \
        --seed "${SEED}" \
        --ad_bound "${AD_BOUND}" \
        --ad_temper_penalty "${penalty}" \
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
        --goal_residual_scale "${GOAL_RESIDUAL_SCALE}" \
        --target_candidate_size "${TARGET_CANDIDATE_SIZE}" \
        --target_projection_mode "${TARGET_PROJECTION_MODE}" \
        --soft_target_tail_advance "${SOFT_TARGET_TAIL_ADVANCE}" \
        --soft_target_popularity_coef "${SOFT_TARGET_POPULARITY_COEF}" \
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
        "${TARGET_MAPPING_ARGS[@]}" \
        "${AGENT_MODE_ARGS[@]}" \
        > "${agent_dir}/log" 2>&1
}

worker() {
    local assigned_slot="$1"
    local current_task_id current_slot gpu gpu_slot environment penalty
    local lambda_fairness lambda_guide subgoal_interval agent_dir
    while IFS=$'\t' read -r current_task_id current_slot gpu gpu_slot \
        environment penalty lambda_fairness lambda_guide subgoal_interval agent_dir
    do
        [[ "${current_task_id}" == "task_id" ]] && continue
        [[ "${current_slot}" != "${assigned_slot}" ]] && continue
        launcher_log="${SWEEP_DIR}/task_${current_task_id}.launcher.log"
        printf '%s START task=%s gpu=%s slot=%s env=%s adp=%s lf=%s lg=%s M=%s\n' \
            "$(date '+%F %T')" "${current_task_id}" "${gpu}" "${gpu_slot}" \
            "${environment}" "${penalty}" "${lambda_fairness}" \
            "${lambda_guide}" "${subgoal_interval}" >> "${DISPATCH_LOG}"
        if run_one "${current_task_id}" "${gpu}" "${environment}" \
            "${penalty}" "${lambda_fairness}" "${lambda_guide}" \
            "${subgoal_interval}" "${agent_dir}" > "${launcher_log}" 2>&1
        then
            printf '%s DONE task=%s gpu=%s env=%s adp=%s lf=%s lg=%s M=%s\n' \
                "$(date '+%F %T')" "${current_task_id}" "${gpu}" \
                "${environment}" "${penalty}" "${lambda_fairness}" \
                "${lambda_guide}" "${subgoal_interval}" >> "${DISPATCH_LOG}"
        else
            exit_code=$?
            printf '%s FAIL task=%s gpu=%s env=%s adp=%s lf=%s lg=%s M=%s exit=%s\n' \
                "$(date '+%F %T')" "${current_task_id}" "${gpu}" \
                "${environment}" "${penalty}" "${lambda_fairness}" \
                "${lambda_guide}" "${subgoal_interval}" "${exit_code}" \
                >> "${DISPATCH_LOG}"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${current_task_id}" "${gpu}" "${environment}" "${penalty}" \
                "${lambda_fairness}" "${lambda_guide}" "${subgoal_interval}" \
                "${exit_code}" >> "${FAILED_FILE}"
        fi
    done < "${TASKS_FILE}"
}

printf 'HRL4PFG sweep: tasks=%d GPUs=%s slots_per_gpu=%d workers=%d dry_run=%s target_mapping=%s max_step=%s seed=%s\n' \
    "${EXPECTED_TASKS}" "${CUDA_DEVICES[*]}" "${SLOTS_PER_GPU}" \
    "${TOTAL_SLOTS}" "${DRY_RUN}" "${TARGET_PROJECTION_MODE}" \
    "${MAX_STEP}" "${SEED}"
printf 'Audit directory: %s\nRun directory: %s\n' "${SWEEP_DIR}" "${RUN_ROOT}"

worker_pids=()
for ((worker_slot = 0; worker_slot < TOTAL_SLOTS; worker_slot++)); do
    worker "${worker_slot}" &
    worker_pids+=("$!")
done

worker_error=0
for worker_pid in "${worker_pids[@]}"; do
    if ! wait "${worker_pid}"; then
        worker_error=1
    fi
done

failed_count="$(awk 'NR > 1 {count++} END {print count + 0}' "${FAILED_FILE}")"
printf 'HRL4PFG sweep finished: tasks=%d failed=%s audit_dir=%s\n' \
    "${EXPECTED_TASKS}" "${failed_count}" "${SWEEP_DIR}"
if (( worker_error != 0 || failed_count != 0 )); then
    exit 1
fi
