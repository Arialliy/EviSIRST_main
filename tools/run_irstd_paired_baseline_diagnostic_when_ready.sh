#!/usr/bin/env bash

set -u
umask 077
# Dynamic descriptors opened with ``exec {name}>>...`` must survive the
# /usr/bin/time -> Python exec chain so the GPU claim remains held end to end.
shopt -u varredir_close

PROJECT_ROOT="/home/ly/EviSIRST_main"
RUN_ROOT="${PROJECT_ROOT}/runs/irstd_performance/complete_target_v1/paired_baseline_diagnostics/run_seed_1446202191"
RUN_LOCK_PATH="${RUN_ROOT}/run.lock"
GPU_LOCK_ROOT="${PROJECT_ROOT}/runs/.gpu_locks"
CHECKPOINT_PATH="${PROJECT_ROOT}/runs/validation_selected/formal/IRSTD-1K/binary/run_seed_1446202191/EviSIRST.pth.tar"
SUMMARY_PATH="${PROJECT_ROOT}/runs/validation_selected/formal/IRSTD-1K/binary/run_seed_1446202191/summary.json"
OUTPUT_PATH="${RUN_ROOT}/matched_target_diagnostics.json"
DIAGNOSTIC="${PROJECT_ROOT}/run_irstd_paired_baseline_diagnostic.py"
BASELINE_TRAINER="${PROJECT_ROOT}/train_validation_selected.py"
PYTHON_BIN="/home/ly/BasicIRSTD/infrarenet/bin/python"
DATASET_ROOT="/home/ly/SCTransNet_main/datasets"
RUN_LOCK_FD_ENV="EVISIRST_PAIRED_DIAGNOSTIC_RUN_LOCK_FD"
GPU_LOCK_FD_ENV="EVISIRST_PAIRED_DIAGNOSTIC_GPU_LOCK_FD"
GPU_UUID_ENV="EVISIRST_PAIRED_DIAGNOSTIC_GPU_UUID"
MEMORY_IDLE_LIMIT_MIB=1024
POLL_SECONDS=10
FAILURE_RETRY_SECONDS=60

timestamp() {
    date '+%Y-%m-%dT%H:%M:%S%z'
}

log() {
    printf '[%s] %s\n' "$(timestamp)" "$*"
}

load_process_argv() {
    local pid="$1"
    local output_name="$2"
    local -n output_ref="${output_name}"
    output_ref=()
    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    mapfile -d '' -t output_ref < "/proc/${pid}/cmdline" 2>/dev/null || return 1
    (( ${#output_ref[@]} > 0 ))
}

argv_has_token() {
    local argv_name="$1"
    local expected="$2"
    local -n argv_ref="${argv_name}"
    local token
    for token in "${argv_ref[@]}"; do
        [[ "${token}" == "${expected}" ]] && return 0
    done
    return 1
}

argv_has_option_value() {
    local argv_name="$1"
    local option="$2"
    local expected="$3"
    local -n argv_ref="${argv_name}"
    local index token
    for (( index=0; index < ${#argv_ref[@]}; index++ )); do
        token="${argv_ref[index]}"
        if [[ "${token}" == "${option}=${expected}" ]]; then
            return 0
        fi
        if [[ "${token}" == "${option}" ]] \
            && (( index + 1 < ${#argv_ref[@]} )) \
            && [[ "${argv_ref[index + 1]}" == "${expected}" ]]; then
            return 0
        fi
    done
    return 1
}

argv_has_script() {
    local pid="$1"
    local argv_name="$2"
    local expected="$3"
    local -n argv_ref="${argv_name}"
    local cwd token candidate
    cwd="$(readlink -f -- "/proc/${pid}/cwd" 2>/dev/null)" || return 1
    for token in "${argv_ref[@]}"; do
        if [[ "${token}" == "${expected}" ]]; then
            return 0
        fi
        if [[ "${token}" != /* ]] && [[ "${token}" == *.py ]]; then
            candidate="$(readlink -f -- "${cwd}/${token}" 2>/dev/null)" || continue
            [[ "${candidate}" == "${expected}" ]] && return 0
        fi
    done
    return 1
}

is_python_process() {
    local pid="$1"
    local executable executable_name
    executable="$(readlink -f -- "/proc/${pid}/exe" 2>/dev/null)" || return 1
    executable_name="${executable##*/}"
    [[ "${executable_name}" == python* ]]
}

baseline_trainer_is_running() {
    local process pid
    local -a argv
    for process in /proc/[0-9]*; do
        pid="${process##*/}"
        is_python_process "${pid}" || continue
        load_process_argv "${pid}" argv || continue
        argv_has_script "${pid}" argv "${BASELINE_TRAINER}" || continue
        argv_has_option_value argv --dataset IRSTD-1K || continue
        argv_has_option_value argv --target-mode binary || continue
        argv_has_option_value argv --run-seed 1446202191 || continue
        argv_has_token argv --smoke && continue
        return 0
    done
    return 1
}

diagnostic_is_running() {
    local process pid
    local -a argv
    for process in /proc/[0-9]*; do
        pid="${process##*/}"
        is_python_process "${pid}" || continue
        load_process_argv "${pid}" argv || continue
        argv_has_script "${pid}" argv "${DIAGNOSTIC}" || continue
        argv_has_option_value argv --dataset-root "${DATASET_ROOT}" || continue
        return 0
    done
    return 1
}

baseline_artifacts_are_ready() {
    [[ -f "${CHECKPOINT_PATH}" && ! -L "${CHECKPOINT_PATH}" ]] \
        && [[ -f "${SUMMARY_PATH}" && ! -L "${SUMMARY_PATH}" ]]
}

# Set QUERY_GPU_INDEX, QUERY_GPU_UUID, QUERY_GPU_BUS, and QUERY_GPU_MEMORY.
query_gpu_identity() {
    local identifier="$1"
    local row raw_index raw_uuid raw_bus raw_memory extra
    row="$(
        nvidia-smi -i "${identifier}" \
            --query-gpu=index,uuid,pci.bus_id,memory.used \
            --format=csv,noheader,nounits 2>/dev/null
    )" || return 1
    IFS=',' read -r raw_index raw_uuid raw_bus raw_memory extra <<< "${row}"
    [[ -z "${extra:-}" ]] || return 1
    QUERY_GPU_INDEX="${raw_index//[[:space:]]/}"
    QUERY_GPU_UUID="${raw_uuid//[[:space:]]/}"
    QUERY_GPU_BUS="${raw_bus//[[:space:]]/}"
    QUERY_GPU_MEMORY="${raw_memory//[[:space:]]/}"
    [[ "${QUERY_GPU_INDEX}" =~ ^[0-9]+$ ]] \
        && [[ "${QUERY_GPU_UUID}" =~ ^GPU-[0-9A-Fa-f-]+$ ]] \
        && [[ "${QUERY_GPU_BUS}" =~ ^[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$ ]] \
        && [[ "${QUERY_GPU_MEMORY}" =~ ^[0-9]+$ ]]
}

gpu_identity_is_idle() {
    local expected_index="$1"
    local expected_uuid="$2"
    local expected_bus="$3"
    local compute_uuids process_count

    # Re-query by immutable UUID, then verify the complete identity tuple.
    query_gpu_identity "${expected_uuid}" || return 1
    [[ "${QUERY_GPU_INDEX}" == "${expected_index}" ]] || return 1
    [[ "${QUERY_GPU_UUID}" == "${expected_uuid}" ]] || return 1
    [[ "${QUERY_GPU_BUS}" == "${expected_bus}" ]] || return 1
    (( QUERY_GPU_MEMORY <= MEMORY_IDLE_LIMIT_MIB )) || return 1
    compute_uuids="$(
        nvidia-smi --query-compute-apps=gpu_uuid \
            --format=csv,noheader,nounits 2>/dev/null
    )" || return 1
    process_count="$(
        printf '%s\n' "${compute_uuids}" \
            | sed 's/[[:space:]]//g' \
            | grep -Fxc -- "${expected_uuid}" || true
    )"
    [[ "${process_count}" == "0" ]]
}

release_selected_gpu_lock() {
    if [[ "${selected_gpu_lock_fd:-}" =~ ^[0-9]+$ ]]; then
        exec {selected_gpu_lock_fd}>&-
    fi
    selected_gpu_lock_fd=""
}

claim_selected_gpu() {
    local gpu_index="$1"
    local gpu_uuid="$2"
    local gpu_bus="$3"
    local lock_path
    lock_path="${GPU_LOCK_ROOT}/${gpu_uuid}.lock"
    [[ ! -L "${lock_path}" ]] || return 1
    selected_gpu_lock_fd=""
    exec {selected_gpu_lock_fd}>>"${lock_path}" || return 1
    if [[ ! -f "${lock_path}" || -L "${lock_path}" ]] \
        || ! flock -n "${selected_gpu_lock_fd}"; then
        release_selected_gpu_lock
        return 1
    fi
    if ! gpu_identity_is_idle "${gpu_index}" "${gpu_uuid}" "${gpu_bus}"; then
        release_selected_gpu_lock
        return 1
    fi
    selected_gpu_index="${gpu_index}"
    selected_gpu_uuid="${gpu_uuid}"
    selected_gpu_bus="${gpu_bus}"
    return 0
}

mkdir -p "${RUN_ROOT}" "${GPU_LOCK_ROOT}"
if [[ -L "${RUN_LOCK_PATH}" ]]; then
    log "fixed run lock is a symbolic link; refusing to start"
    exit 1
fi
exec 9>>"${RUN_LOCK_PATH}"
if [[ ! -f "${RUN_LOCK_PATH}" || -L "${RUN_LOCK_PATH}" ]]; then
    log "fixed run lock is not a regular file; refusing to start"
    exit 1
fi
if ! flock -n 9; then
    log "another paired-baseline diagnostic or watcher holds ${RUN_LOCK_PATH}; exiting"
    exit 0
fi
export "${RUN_LOCK_FD_ENV}=9"

declare -A idle_streak=()
last_status_epoch=0
selected_gpu_index=""
selected_gpu_uuid=""
selected_gpu_bus=""
selected_gpu_lock_fd=""
cd "${PROJECT_ROOT}" || exit 1
log "watcher started; waiting for the completed paired R1 baseline"

while true; do
    if [[ -f "${OUTPUT_PATH}" && ! -L "${OUTPUT_PATH}" ]]; then
        log "diagnostic output already exists; refusing to overwrite and exiting"
        exit 0
    fi
    if [[ -e "${OUTPUT_PATH}" || -L "${OUTPUT_PATH}" ]]; then
        log "diagnostic output path exists but is not a regular file; refusing to start"
        exit 1
    fi

    if baseline_artifacts_are_ready && ! baseline_trainer_is_running; then
        if diagnostic_is_running; then
            log "a lock-bypassing paired-baseline diagnostic is running; monitoring continues"
            sleep "${POLL_SECONDS}"
            continue
        fi

        mapfile -t gpu_indices < <(
            nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null \
                | sed 's/[[:space:]]//g' \
                | grep -E '^[0-9]+$'
        )
        selected_gpu_index=""
        selected_gpu_uuid=""
        selected_gpu_bus=""
        for gpu_index in "${gpu_indices[@]}"; do
            if query_gpu_identity "${gpu_index}"; then
                gpu_uuid="${QUERY_GPU_UUID}"
                gpu_bus="${QUERY_GPU_BUS}"
            else
                continue
            fi
            if gpu_identity_is_idle "${gpu_index}" "${gpu_uuid}" "${gpu_bus}"; then
                idle_streak["${gpu_uuid}"]=$(( ${idle_streak["${gpu_uuid}"]:-0} + 1 ))
                if (( idle_streak["${gpu_uuid}"] >= 2 )) \
                    && claim_selected_gpu "${gpu_index}" "${gpu_uuid}" "${gpu_bus}"; then
                    break
                fi
            else
                idle_streak["${gpu_uuid}"]=0
            fi
        done

        if [[ -n "${selected_gpu_uuid}" ]]; then
            if [[ -e "${OUTPUT_PATH}" || -L "${OUTPUT_PATH}" ]] \
                || diagnostic_is_running \
                || baseline_trainer_is_running \
                || ! baseline_artifacts_are_ready \
                || ! gpu_identity_is_idle \
                    "${selected_gpu_index}" "${selected_gpu_uuid}" "${selected_gpu_bus}"; then
                log "launch preconditions changed; releasing GPU claim and continuing"
                idle_streak["${selected_gpu_uuid}"]=0
                release_selected_gpu_lock
                sleep "${POLL_SECONDS}"
                continue
            fi

            log "baseline complete; GPU index=${selected_gpu_index} bus=${selected_gpu_bus} uuid=${selected_gpu_uuid} is idle and cooperatively locked"
            export CUDA_DEVICE_ORDER=PCI_BUS_ID
            export CUDA_VISIBLE_DEVICES="${selected_gpu_uuid}"
            export PYTHONDONTWRITEBYTECODE=1
            export PYTHONUNBUFFERED=1
            export "${GPU_LOCK_FD_ENV}=${selected_gpu_lock_fd}"
            export "${GPU_UUID_ENV}=${selected_gpu_uuid}"
            diagnostic_status=0
            /usr/bin/time -v "${PYTHON_BIN}" "${DIAGNOSTIC}" \
                --dataset-root "${DATASET_ROOT}" \
                --device cuda:0 \
                --workers 0 || diagnostic_status=$?
            release_selected_gpu_lock
            unset CUDA_VISIBLE_DEVICES
            unset "${GPU_LOCK_FD_ENV}"
            unset "${GPU_UUID_ENV}"

            if [[ -f "${OUTPUT_PATH}" && ! -L "${OUTPUT_PATH}" ]]; then
                if (( diagnostic_status == 0 )); then
                    log "paired-baseline diagnostic completed; immutable output exists"
                    exit 0
                fi
                log "diagnostic failed with exit status ${diagnostic_status} after an immutable output appeared; refusing to retry or overwrite"
                exit 1
            fi
            if [[ -e "${OUTPUT_PATH}" || -L "${OUTPUT_PATH}" ]]; then
                log "diagnostic output path appeared but is not a regular file; refusing to retry"
                exit 1
            fi
            log "diagnostic failed with exit status ${diagnostic_status} and no output; retrying after ${FAILURE_RETRY_SECONDS}s"
            idle_streak["${selected_gpu_uuid}"]=0
            sleep "${FAILURE_RETRY_SECONDS}"
            continue
        fi
    fi

    now_epoch="$(date +%s)"
    if (( now_epoch - last_status_epoch >= 300 )); then
        if baseline_artifacts_are_ready; then
            log "baseline artifacts exist, but trainer/GPU readiness is pending"
        else
            log "baseline is incomplete; monitoring continues"
        fi
        last_status_epoch="${now_epoch}"
    fi
    sleep "${POLL_SECONDS}"
done
