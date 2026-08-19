#!/usr/bin/env python3
"""Fail-closed idle-GPU watcher for the fixed IRSTD A/B legacy screen.

The seven eventual target tasks are immutable, but this revision authorizes
only the two Seed-42 Stage-1 candidates.  The existing clean R1 Seed-42
first-500 evidence is reused.  Clean run-seed 104728269 and both candidates'
1446202191/104728269 trajectories stay locked until separately frozen S1
decision artifacts exist.  Every worker keeps the formal 1000-epoch schedule.
The watcher wraps the transaction engine's post-commit candidate
validation/cleanup and blocks only after epoch 500 is completely committed;
it then terminates the worker's independent process group before epoch 501.

This watcher is standard-library-only.  Torch and all experiment modules are
loaded only by short fixed-interpreter child processes.  Formal launch is
gated by a reviewed authorization artifact and its frozen SHA-256.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


PROJECT_ROOT = Path("/home/ly/EviSIRST_main")
PYTHON_BIN = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
EXPECTED_PYTHON = Path("/usr/bin/python3.12")
DATASET_ROOT = Path("/home/ly/SCTransNet_main/datasets")
SPLIT_ROOT = PROJECT_ROOT / "splits/v2"
CLEAN_TRAINER = PROJECT_ROOT / "train_validation_selected.py"
PSBFR_TRAINER = PROJECT_ROOT / "train_irstd_model_design_screen_v1.py"
CP_HF_S2_TRAINER = PROJECT_ROOT / "train_irstd_cp_hf_s2_legacy_screen_v1.py"
STAGE1_CONTROL_READER = PROJECT_ROOT / "run_irstd_model_design_epoch500_gate_v1.py"
WATCHER_SOURCE = PROJECT_ROOT / "tools/run_irstd_ab_model_screen_when_idle.py"
TIME_BIN = Path("/usr/bin/time")
NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
PROC_ROOT = Path("/proc")

WATCH_ROOT = PROJECT_ROOT / "runs/irstd_model_design/ab_legacy_screen_watcher_v1"
WATCHER_LOCK = WATCH_ROOT / "watcher.lock"
TASK_LOCK_ROOT = WATCH_ROOT / "task_locks"
LOG_ROOT = WATCH_ROOT / "logs"
COMPLETION_ROOT = WATCH_ROOT / "epoch500_commits"
GPU_LOCK_ROOT = PROJECT_ROOT / "runs/.gpu_locks"

AUTHORIZATION_PATH = (
    PROJECT_ROOT / "experiments/IRSTD_AB_MODEL_SCREEN_V1_AUTHORIZATION.json"
)
# Reviewed Stage-1-only authorization; creating the artifact does not itself
# launch a worker.  Only the explicit ``--formal`` mode can enter supervision.
FORMAL_AUTHORIZATION_SHA256: str | None = "e99d99d7a6207f7aac05ed63ab760a0212dbd36227d0f6ee741d93ed48b42d1a"
AUTHORIZATION_SCHEMA = "evisirst_irstd_ab_model_screen_authorization/v1"
COMPLETION_SCHEMA = "evisirst_irstd_ab_model_screen_epoch500_stop/v1"

FINAL_CONFIGURED_EPOCH = 1000
STOP_EPOCH = 500
CLEAN_INITIAL_EPOCH = 464
POLL_SECONDS = 10
GPU_CONFIRM_SECONDS = 10
GPU_MEMORY_LIMIT_MIB = 1024
TERM_TIMEOUT_SECONDS = 60
_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PROBE_PREFIX = "EVISIRST_AB_MODEL_SCREEN_PROBE:"
_CONTROL_PROBE_PREFIX = "EVISIRST_AB_MODEL_SCREEN_CONTROL_PROBE:"


class ABModelScreenWatcherError(RuntimeError):
    """The frozen A/B queue cannot be supervised safely."""


@dataclass(frozen=True)
class Task:
    name: str
    kind: str
    variant: str
    run_seed: int
    run_dir: Path
    minimum_resume_epoch: int
    fresh_allowed: bool


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    bus_id: str
    memory_used_mib: int


@dataclass
class RunningJob:
    task: Task
    process: subprocess.Popen[bytes]
    gpu: GPU
    gpu_lock_fd: int
    task_lock_fd: int
    log_handle: object
    log_path: Path
    log_start_offset: int


def _design_run_dir(variant: str, seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "runs/irstd_model_design"
        / variant
        / "legacy_screen/formal/IRSTD-1K/binary"
        / f"run_seed_{seed}"
    )


TASKS = (
    Task(
        "clean_r1_s104728269_to500",
        "clean",
        "clean_evisirst",
        104_728_269,
        PROJECT_ROOT
        / "runs/validation_selected/formal/IRSTD-1K/binary/run_seed_104728269",
        CLEAN_INITIAL_EPOCH,
        False,
    ),
    Task(
        "psbfr_s42",
        "psbfr",
        "psbfr_v1",
        42,
        _design_run_dir("psbfr_v1", 42),
        1,
        True,
    ),
    Task(
        "psbfr_s1446202191",
        "psbfr",
        "psbfr_v1",
        1_446_202_191,
        _design_run_dir("psbfr_v1", 1_446_202_191),
        1,
        True,
    ),
    Task(
        "psbfr_s104728269",
        "psbfr",
        "psbfr_v1",
        104_728_269,
        _design_run_dir("psbfr_v1", 104_728_269),
        1,
        True,
    ),
    Task(
        "cp_hf_s2_s42",
        "cp_hf_s2",
        "cp_hf_s2_v1",
        42,
        _design_run_dir("cp_hf_s2_v1", 42),
        1,
        True,
    ),
    Task(
        "cp_hf_s2_s1446202191",
        "cp_hf_s2",
        "cp_hf_s2_v1",
        1_446_202_191,
        _design_run_dir("cp_hf_s2_v1", 1_446_202_191),
        1,
        True,
    ),
    Task(
        "cp_hf_s2_s104728269",
        "cp_hf_s2",
        "cp_hf_s2_v1",
        104_728_269,
        _design_run_dir("cp_hf_s2_v1", 104_728_269),
        1,
        True,
    ),
)

# The authoritative plan requires a Seed-42 engineering gate before any other
# trajectory.  No fixed S1 decision artifact exists yet, so this revision is
# deliberately a Stage-1-only launcher.  The other five immutable target tasks
# remain in the manifest but cannot be converted into worker commands or
# launched by this source.
STAGE1_TASK_NAMES = frozenset({"psbfr_s42", "cp_hf_s2_s42"})
STAGE2_LOCKED_TASK_NAMES = frozenset(
    task.name for task in TASKS if task.name not in STAGE1_TASK_NAMES
)
STAGE1_CLEAN_CONTROL_RUN_DIR = (
    PROJECT_ROOT
    / "runs/validation_selected/formal/IRSTD-1K/binary/run_seed_42"
)
FROZEN_STAGE1_CLEAN_CONTROL_SHA256 = {
    "last_training_state.pth.tar": (
        "68dbb6f5ee303682a5987fb32ad91a1d819885f8eafe942765fd20b95609f9c8"
    ),
    "validation_history.json": (
        "3881fedb39c811ec382991833b1ff04e3199242e9d89eaae845ea0a3fcf7dace"
    ),
}
FROZEN_STAGE2_CLEAN_RESUME_EPOCH = CLEAN_INITIAL_EPOCH
FROZEN_STAGE2_CLEAN_RESUME_SHA256 = {
    "last_training_state.pth.tar": (
        "2fa19677d92c646bdbe721cbe3d31410bdd51e027ef7c040107f640db3c5d2af"
    ),
    "validation_history.json": (
        "612e04cb8635c2bee6d81166159bfd189da74a7c654469321c7753b0307063d8"
    ),
}
FROZEN_CP_FRESH_SOURCE_TREE_SHA256 = (
    "023388b3a7931f9da313185e187161de99b956b5e958c46a9edc3535f10bc0cb"
)


def _task_phase(task: Task) -> str:
    if task.name in STAGE1_TASK_NAMES:
        return "S1_seed42"
    if task.kind == "clean":
        return "S2_shared_clean_control"
    return "S2_route_confirmation"


# Exhaustive runtime/evidence source set frozen after the independent CP-HF-S2
# Stage-1 audit.  Any byte change disables preflight and formal launch.
FROZEN_SOURCE_SHA256: dict[str, str] = {
    "experiments/IRSTD_CP_HF_S2_LEGACY_SCREEN_V1_AMENDMENT.md": (
        "0ea707ef546b8154eecdcaf232d20f9a28673dba8ce6baa4cbb583e6dfc67d57"
    ),
    "experiments/IRSTD_PSBFR_V1_PROTOCOL.md": (
        "61bfed5a23d57d0aec0bdb8690ecc3f81e8fc1d7d775c8f947977b3f02db9640"
    ),
    "experiments/evisirst_data.py": (
        "93f004cfbd0ed80cb04a76728d7cae4bc4317c011a1f31e2ff38ecd26589a3f1"
    ),
    "experiments/evisirst_v2_data.py": (
        "9c6cfabf33c9c16a99574da6d87c9318800b981e585a5603ce257250fbc2b620"
    ),
    "experiments/evisirst_v2_selection.py": (
        "763a027abdfae79726faf31fe479801c82642d08c8194566029fe818153d575c"
    ),
    "experiments/evisirst_v2_splits.py": (
        "e6805f9c27106982effdc36bb8965836ae14625f51fe39a2a10816a3d030b1dc"
    ),
    "experiments/evisirst_zero_margin_selection.py": (
        "776afca34eb5a83f514d145186fed8910548e902a9690620eefa495b94ec94d8"
    ),
    "experiments/four_dataset_models_seed42_v1.py": (
        "a7127fc334ea72b2021aa670f341b0d67a6070e82b9e780bd5c56ed555d0a4d3"
    ),
    "experiments/irstd_hf_decoder_v1.py": (
        "e669beaeae7606e5dae96e6137d9eee9eb8fd27459432c0fa6ea607816779502"
    ),
    "experiments/irstd_cp_hf_s2_legacy_screen_v1_rules.json": (
        "da811d143623fb7d5265296cb16c45b2475e0d8478f9faad835d72972878b603"
    ),
    "experiments/irstd_cp_hf_s2_v1.py": (
        "0dba0021dfe8f46d8c4e2932b478ce03609ab14be73b20c8b77817dce240436a"
    ),
    "experiments/irstd_psbfr_v1.py": (
        "93c2877570ad699055f669e6ea43ab71b4f9b7d8b29132e49ceca96d7e6e7671"
    ),
    "experiments/irstd_psbfr_v1_screen_rules.json": (
        "7d08b877a8683ae1c7ba63139b48d26c9b6be329045edea5af924ada1c1e0b49"
    ),
    "experiments/irstd_single_residual_v1.py": (
        "30708a072296ef097b229ea63a0a3f60c3a163b42f371df85ef547689a04710b"
    ),
    "experiments/three_dataset_v2_protocol.py": (
        "c6aa18250e23f718a4284de0eff8862edff9339d092398d0a6f84a523d0495be"
    ),
    "model/EviSIRST.py": (
        "09c3f74910eb2ee3ddd454546bf3e21e7c463a8ff2740514c8f4f3c640204b55"
    ),
    "model/__init__.py": (
        "e9a5828ff0d7933df343c3e621e9cb46e69ddb1182afc97d389e5363cb419c2e"
    ),
    "model/_internal/Config.py": (
        "b7e3e67c379ef4638605ebe612336b0c3cdb1a97f4d6fe731dec80b4847d5596"
    ),
    "model/_internal/SCTransNet.py": (
        "5fb7ce711f190ead2bfcc910d2971266b2561e643c9f8a524d2032ffd48c0aeb"
    ),
    "model/_internal/tpd_clean.py": (
        "ca61b4035bc32c99d5862e08c48096e4892e90c0de1d1f3bdf12d683c94597f2"
    ),
    "model/_internal/tpd_clean_v8_mprs_dch.py": (
        "39e7b1618ea9f594f4abecc284afd0d845132218e2fab3657dbb957c78703c19"
    ),
    "model/_internal/tpd_forward_contract.py": (
        "136dc2fc91276f78cfce0eb5a55e61e7bacb904ffae326bd2b61aa2fc357a94f"
    ),
    "model/_internal/tpd_frequency_gate.py": (
        "5290bf2fd13516ea0726103dd3661fbf1ef585098b8fe933f38972887a78f4a5"
    ),
    "model/_internal/tpd_frequency_gate_v2_croa.py": (
        "38a60bc43c75029e85ede5041420a8aae48dbba30aa4ae3a9470d31136281002"
    ),
    "model/_internal/tpd_ner_v8_mprs_dch.py": (
        "d0d9dd226c64689f118b5d179c5415a82d26a6c803337a0f02aaaba789491acd"
    ),
    "model/_internal/tpd_ner_v8_mprs_dch_v2.py": (
        "b6a666b00d9f53ad153089a80861907e579ec97398c66a542d20cffded03fa6b"
    ),
    "model/_internal/tpd_ner_v8_mprs_dch_v3.py": (
        "f8fd2e307e29fd6ff26d15a7b7a1daaee22e245bc2721d68c29d34e7c653d291"
    ),
    "model/_internal/tpd_ner_v8_mprs_dch_v4_tail_aware.py": (
        "ce64dbbdfba76cdf8f3f2f331e1ac60d4e76e23a053d73a83b785e9ca1edf37f"
    ),
    "model/_internal/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py": (
        "c8c7db1fc8b3e83c45ee11dcb45f7c09dd2f4456554c4267aaba3b369394ff53"
    ),
    "model/_internal/tpd_ner_v8_mprs_dch_v4_tail_aware_survival.py": (
        "3ff5c5a383384c727ea2eeb628b6121ee54f41c3cfa90362c411b7e70236703a"
    ),
    "model/_internal/tpd_query_frequency_bridge.py": (
        "a8a6cbe18917590349e8d354d1966382414175fccabc09225c342627b975e580"
    ),
    "model/_internal/tpd_relay.py": (
        "7b69d8e7cc62a5883355c516bdc0d30ca37036613d96d4a8ba54f51ee8744345"
    ),
    "model/_internal/tpd_sctransnet.py": (
        "cf887b55411da8219f32b8c2a5e248b5ffec403f13146a6a1806fec4e1c014af"
    ),
    "model/_internal/tpd_survival.py": (
        "32903ddab6da5ebc1cc941240890bbd4c4c5dcbcd34515a4b36ccd9ddbf33e40"
    ),
    "run_irstd_cp_hf_s2_first500_ab_gate_v1.py": (
        "9250b7d0de5ef04b08a14bb0e931a7b2034ef0056c3262c13b247783038907d1"
    ),
    "run_irstd_cp_hf_s2_mechanism_diagnostic_v1.py": (
        "ffcefbfa1a582760a9838a68e009aaa41e7a60de469c135b921eeb57e802e4c0"
    ),
    "run_irstd_cp_hf_s2_stage1_route_gate_v1.py": (
        "690679e5f6a61c6d8365df3f5d2aa10bf174b8ee70260e2801edd55305844d40"
    ),
    "run_irstd_model_design_epoch500_gate_v1.py": (
        "e67bb7bf39bca9d30782bc76bb01abebf11fd7d550bca5dcb56e2a16762b199c"
    ),
    "run_irstd_psbfr_correction_diagnostic_v1.py": (
        "b77416316d3ca9c4d871c4c9be9e941f68b942788aa9e980ea2078ccec4172a0"
    ),
    "splits/v2/IRSTD-1K/manifest.json": (
        "5eeacbab52d8b70b44ce4cb70dfeda94cd326767f0c379fa96ab836c4c897596"
    ),
    "splits/v2/IRSTD-1K/train.txt": (
        "460083baae2ba23f5629e7bd346b5623f78a96856c83693b96bf917f6537ed2d"
    ),
    "splits/v2/IRSTD-1K/val.txt": (
        "05a0d0ecdb1772447c4b5b0b04a5e8cf3a748ba5fdab5cdb3f9323d9089e9576"
    ),
    "tests/test_irstd_ab_model_screen_watcher.py": (
        "ed0dc598d53d4193e04d9e7e8131e524b6a7d43c40154d661efbf62592697adc"
    ),
    "tests/test_irstd_cp_hf_s2_first500_ab_gate_v1.py": (
        "46d4524eea29d882d92f56936823f3f895195d06d1ccb036c5da7dee6287cf49"
    ),
    "tests/test_irstd_cp_hf_s2_mechanism_diagnostic_v1.py": (
        "4bdef0fd93910ebdda3c0f793cd669070700364b45d4f1c6c74cc3dc9fdc3d09"
    ),
    "tests/test_irstd_cp_hf_s2_stage1_route_gate_v1.py": (
        "d3fa9b4ffcfd15abd8091061d2552c35a0748453a9d51849d5f7b55e38cbe9e0"
    ),
    "tests/test_irstd_cp_hf_s2_v1.py": (
        "c4add1293f6cbd9924993277f01bc49b26e1dc3f87ffad7148de9aaf1dfa6394"
    ),
    "tests/test_irstd_model_design_epoch500_gate_v1.py": (
        "5bba448fed7a753bbef23ec6347ba3964821acfec43186e76a79a335be0e395d"
    ),
    "tests/test_irstd_psbfr_correction_diagnostic_v1.py": (
        "df7eb1edfd182d3d8edbe3add413d9d3b09795533ec6ba95e94a13350a9a6ed9"
    ),
    "tests/test_train_irstd_cp_hf_s2_legacy_screen_v1.py": (
        "cd2a80194c141b411d392bee5ce937d5a8307dce8ca93d91679d26334b60bce2"
    ),
    "tests/test_train_irstd_model_design_screen_v1.py": (
        "99f19001138721d20f223b9fe2d309da34365741808ce2d59d175daf8e32476b"
    ),
    "train.py": (
        "93b2c537e127c1363429516eb616c4874613703e12fe083737191e19dbf2925e"
    ),
    "train_irstd_cp_hf_s2_legacy_screen_v1.py": (
        "0383b5f2769b510b7040639a8cfdd0209cc91b901062b2f3d9eb897eb1efedab"
    ),
    "train_irstd_hf_decoder_v1.py": (
        "4e0a576d604812bc9b3369bd6065459d4c0f814010a9157665f201400a173eb5"
    ),
    "train_irstd_model_design_screen_v1.py": (
        "9d743f29c3b080aa03fd5522cd3f0c865bc829ef422ca4985f2ce6030d3afd4a"
    ),
    "train_validation_selected.py": (
        "6f870677f7724054fc8bc71e07ddde46ed3d316eb837f953be50fe8c99c6e0ed"
    ),
}


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ABModelScreenWatcherError("value is not strict canonical JSON") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _watcher_source_sha256() -> str:
    """Hash watcher code while normalizing only the authorization digest.

    The authorization binds this code hash, while this file binds the
    authorization hash.  Normalizing that single literal removes the otherwise
    circular hash dependency without excluding any executable watcher logic.
    """

    source = WATCHER_SOURCE.read_bytes()
    pattern = re.compile(
        rb"FORMAL_AUTHORIZATION_SHA256: str \| None = "
        rb"(?:None|\"[0-9a-f]{64}\")"
    )
    normalized, count = pattern.subn(
        b'FORMAL_AUTHORIZATION_SHA256: str | None = "<AUTHORIZATION_SHA256>"',
        source,
    )
    if count != 1:
        raise ABModelScreenWatcherError(
            "watcher authorization literal normalization differs"
        )
    return hashlib.sha256(normalized).hexdigest()


def _regular_project_file(relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise ABModelScreenWatcherError("source allowlist path is unsafe")
    current = PROJECT_ROOT
    for component in candidate.parts:
        current /= component
        if current.is_symlink():
            raise ABModelScreenWatcherError(f"source path is a symlink: {relative}")
    if not current.is_file():
        raise ABModelScreenWatcherError(f"source is not a regular file: {relative}")
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(PROJECT_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ABModelScreenWatcherError("source escaped the project") from exc
    return resolved


def assert_frozen_sources() -> None:
    if not FROZEN_SOURCE_SHA256:
        raise ABModelScreenWatcherError("source SHA-256 allowlist is empty")
    for relative, expected in FROZEN_SOURCE_SHA256.items():
        if not _SHA256_RE.fullmatch(expected):
            raise ABModelScreenWatcherError(
                f"source SHA-256 is not frozen: {relative}"
            )
        if _sha256_file(_regular_project_file(relative)) != expected:
            raise ABModelScreenWatcherError(f"source SHA-256 differs: {relative}")


def frozen_sources_are_valid() -> bool:
    try:
        assert_frozen_sources()
    except (ABModelScreenWatcherError, OSError):
        return False
    return True


def frozen_source_allowlist_sha256() -> str:
    return _canonical_sha256(FROZEN_SOURCE_SHA256)


def _assert_safe_task_run_chain(task: Task) -> None:
    runs = PROJECT_ROOT / "runs"
    try:
        relative = task.run_dir.relative_to(runs)
    except ValueError as exc:
        raise ABModelScreenWatcherError("task run directory escaped runs") from exc
    current = runs
    if current.is_symlink() or (current.exists() and not current.is_dir()):
        raise ABModelScreenWatcherError("repository runs root is unsafe")
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ABModelScreenWatcherError(
                f"task run directory chain contains a symlink: {current}"
            )
        if current.exists() and not current.is_dir():
            raise ABModelScreenWatcherError(
                f"task run directory chain contains a non-directory: {current}"
            )
        if not current.exists():
            break


def assert_frozen_stage1_clean_control_artifacts() -> None:
    for name, expected in FROZEN_STAGE1_CLEAN_CONTROL_SHA256.items():
        if not _SHA256_RE.fullmatch(expected):
            raise ABModelScreenWatcherError("clean Seed-42 artifact SHA is malformed")
        path = STAGE1_CLEAN_CONTROL_RUN_DIR / name
        if path.is_symlink() or not path.is_file():
            raise ABModelScreenWatcherError(
                f"clean Seed-42 artifact is not regular: {name}"
            )
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(STAGE1_CLEAN_CONTROL_RUN_DIR.resolve(strict=True))
        except ValueError as exc:
            raise ABModelScreenWatcherError(
                "clean Seed-42 artifact escaped its run directory"
            ) from exc
        if _sha256_file(resolved) != expected:
            raise ABModelScreenWatcherError(
                f"clean Seed-42 artifact SHA differs: {name}"
            )


def assert_frozen_stage2_clean_resume_artifacts() -> None:
    task = next(item for item in TASKS if item.name == "clean_r1_s104728269_to500")
    if task.minimum_resume_epoch != FROZEN_STAGE2_CLEAN_RESUME_EPOCH:
        raise ABModelScreenWatcherError("locked clean resume epoch differs")
    _assert_safe_task_run_chain(task)
    for name, expected in FROZEN_STAGE2_CLEAN_RESUME_SHA256.items():
        if not _SHA256_RE.fullmatch(expected):
            raise ABModelScreenWatcherError("locked clean artifact SHA is malformed")
        path = task.run_dir / name
        if path.is_symlink() or not path.is_file():
            raise ABModelScreenWatcherError(
                f"locked clean artifact is not regular: {name}"
            )
        if _sha256_file(path) != expected:
            raise ABModelScreenWatcherError(
                f"locked clean artifact SHA differs: {name}"
            )


def stage2_clean_resume_is_frozen() -> bool:
    try:
        assert_frozen_stage2_clean_resume_artifacts()
    except (ABModelScreenWatcherError, OSError):
        return False
    return True


def _trainer_for(task: Task) -> Path:
    if task.kind == "clean":
        return CLEAN_TRAINER
    if task.kind == "psbfr":
        return PSBFR_TRAINER
    if task.kind == "cp_hf_s2":
        return CP_HF_S2_TRAINER
    raise ABModelScreenWatcherError(f"unknown task kind: {task.kind}")


def formal_training_argv(task: Task, *, resume: bool) -> tuple[str, ...]:
    if task.kind == "clean":
        if not resume:
            raise ABModelScreenWatcherError("clean R1 is a resume-only task")
        argv = (
            os.fspath(PYTHON_BIN),
            os.fspath(CLEAN_TRAINER),
            "--dataset",
            "IRSTD-1K",
            "--dataset-root",
            os.fspath(DATASET_ROOT),
            "--split-root",
            os.fspath(SPLIT_ROOT),
            "--output-root",
            os.fspath(PROJECT_ROOT / "runs/validation_selected"),
            "--target-mode",
            "binary",
            "--architecture-seed",
            "42",
            "--run-seed",
            str(task.run_seed),
            "--device",
            "cuda:0",
            "--epochs",
            "1000",
            "--batch-size",
            "16",
            "--workers",
            "0",
            "--base-lr",
            "0.001",
            "--min-lr",
            "0.00001",
            "--warmup-epochs",
            "10",
            "--val-interval",
            "1",
            "--allow-sample-level-fallback",
        )
    elif task.kind == "psbfr":
        argv = (
            os.fspath(PYTHON_BIN),
            os.fspath(PSBFR_TRAINER),
            "--dataset-root",
            os.fspath(DATASET_ROOT),
            "--split-root",
            os.fspath(SPLIT_ROOT),
            "--variant",
            "psbfr_v1",
            "--dataset",
            "IRSTD-1K",
            "--target-mode",
            "binary",
            "--architecture-seed",
            "42",
            "--run-seed",
            str(task.run_seed),
            "--device",
            "cuda:0",
            "--epochs",
            "1000",
            "--warmup-epochs",
            "10",
            "--allow-sample-level-fallback",
        )
    elif task.kind == "cp_hf_s2":
        argv = (
            os.fspath(PYTHON_BIN),
            os.fspath(CP_HF_S2_TRAINER),
            "--dataset-root",
            os.fspath(DATASET_ROOT),
            "--run-seed",
            str(task.run_seed),
            "--device",
            "cuda:0",
            "--epochs",
            "1000",
            "--warmup-epochs",
            "10",
        )
    else:
        raise ABModelScreenWatcherError(f"unknown task kind: {task.kind}")
    return argv + (("--resume",) if resume else ())


def timed_argv(argv: Sequence[str]) -> tuple[str, ...]:
    return os.fspath(TIME_BIN), "-v", *tuple(argv)


def task_manifest() -> dict[str, object]:
    entries = []
    for task in TASKS:
        entries.append(
            {
                "name": task.name,
                "kind": task.kind,
                "variant": task.variant,
                "run_seed": task.run_seed,
                "run_dir": task.run_dir.relative_to(PROJECT_ROOT).as_posix(),
                "minimum_resume_epoch": task.minimum_resume_epoch,
                "fresh_allowed": task.fresh_allowed,
                "phase": _task_phase(task),
                "launch_authorized_by_this_watcher": (
                    task.name in STAGE1_TASK_NAMES
                ),
                "runner": _trainer_for(task).relative_to(PROJECT_ROOT).as_posix(),
                "fresh_argv": (
                    list(formal_training_argv(task, resume=False))
                    if task.fresh_allowed
                    else None
                ),
                "resume_argv": list(formal_training_argv(task, resume=True)),
            }
        )
    return {
        "schema": "evisirst_irstd_ab_model_screen_task_manifest/v1",
        "configured_total_epochs": FINAL_CONFIGURED_EPOCH,
        "controlled_stop_epoch": STOP_EPOCH,
        "launcher_stage": "S1_seed42_only",
        "s2_unlock_implemented": False,
        "s2_unlock_reason": "fixed_route_specific_s1_decision_artifacts_unavailable",
        "stage1_task_names": sorted(STAGE1_TASK_NAMES),
        "stage2_locked_task_names": sorted(STAGE2_LOCKED_TASK_NAMES),
        "stage1_clean_control": {
            "architecture_seed": 42,
            "run_seed": 42,
            "screen_prefix_epochs": STOP_EPOCH,
            "run_dir": STAGE1_CLEAN_CONTROL_RUN_DIR.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "artifact_sha256": dict(FROZEN_STAGE1_CLEAN_CONTROL_SHA256),
            "reader": STAGE1_CONTROL_READER.relative_to(PROJECT_ROOT).as_posix(),
        },
        "stage2_locked_clean_resume": {
            "task": "clean_r1_s104728269_to500",
            "committed_epoch": FROZEN_STAGE2_CLEAN_RESUME_EPOCH,
            "artifact_sha256": dict(FROZEN_STAGE2_CLEAN_RESUME_SHA256),
        },
        "cp_fresh_source_tree_sha256": FROZEN_CP_FRESH_SOURCE_TREE_SHA256,
        "candidate_decisions_are_independent": True,
        "candidate_ranking_or_winner_selection": False,
        "source_allowlist_sha256": frozen_source_allowlist_sha256(),
        "tasks": entries,
        "lockbox_accessed": False,
        "public_test_allowed": False,
        "test_split_accessed": False,
    }


def _strict_json(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise ABModelScreenWatcherError(f"not a regular JSON file: {path}")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ABModelScreenWatcherError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ABModelScreenWatcherError(f"non-finite JSON constant: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ABModelScreenWatcherError("JSON artifact is malformed") from exc

    def finite(item: object) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ABModelScreenWatcherError("non-finite JSON number")
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)

    finite(value)
    return value


def authorization_is_valid() -> bool:
    expected = FORMAL_AUTHORIZATION_SHA256
    if expected is None:
        return False
    if not _SHA256_RE.fullmatch(expected):
        raise ABModelScreenWatcherError("authorization SHA-256 is malformed")
    try:
        assert_frozen_sources()
        assert_frozen_stage1_clean_control_artifacts()
        assert_frozen_stage2_clean_resume_artifacts()
        if _sha256_file(AUTHORIZATION_PATH) != expected:
            return False
        payload = _strict_json(AUTHORIZATION_PATH)
        watcher_sha = _watcher_source_sha256()
    except (ABModelScreenWatcherError, OSError):
        return False
    required_keys = {
        "schema",
        "status",
        "task_manifest_sha256",
        "watcher_source_sha256",
        "source_allowlist_sha256",
        "lockbox_accessed",
        "public_test_allowed",
        "test_split_accessed",
    }
    return bool(
        isinstance(payload, dict)
        and set(payload) == required_keys
        and payload.get("schema") == AUTHORIZATION_SCHEMA
        and payload.get("status") == "AUTHORIZED"
        and payload.get("task_manifest_sha256")
        == _canonical_sha256(task_manifest())
        and payload.get("watcher_source_sha256") == watcher_sha
        and payload.get("source_allowlist_sha256")
        == frozen_source_allowlist_sha256()
        and payload.get("lockbox_accessed") is False
        and payload.get("public_test_allowed") is False
        and payload.get("test_split_accessed") is False
    )


def make_cleanup_barrier(
    original: Callable[..., object],
    *,
    pause: Callable[[], object],
    announce: Callable[[], object],
) -> Callable[..., object]:
    """Block after epoch-500 commit cleanup and reject any observed epoch 501."""

    def wrapped(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        epoch = kwargs.get("completed_epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise ABModelScreenWatcherError("cleanup completed_epoch is malformed")
        if epoch > STOP_EPOCH:
            raise ABModelScreenWatcherError("epoch 501 is forbidden in A/B screen")
        if epoch == STOP_EPOCH:
            announce()
            while True:
                pause()
        return result

    return wrapped


_FORMAL_WORKER_PROGRAM = r'''
import signal, sys

kind=sys.argv[1]
seed=int(sys.argv[2])
resume=sys.argv[3] == "resume"

from tools import run_irstd_ab_model_screen_when_idle as watcher
watcher.assert_frozen_sources()
if not watcher.authorization_is_valid():
    raise RuntimeError("formal A/B authorization is unavailable")
if not watcher.stage1_clean_control_is_valid():
    raise RuntimeError("clean Seed-42 first-500 control is unavailable")
matches=[task for task in watcher.TASKS
         if task.kind == kind and task.run_seed == seed]
if len(matches) != 1 or matches[0].name not in watcher.STAGE1_TASK_NAMES:
    raise RuntimeError("worker request is outside the Stage-1 authorization")
expected_states={"resume","ready"} if resume else {"fresh"}
if watcher._probe_task_state(matches[0]) not in expected_states:
    raise RuntimeError("worker launch state changed after scheduling")

def announce():
    print("EVISIRST_AB_MODEL_SCREEN_WORKER:EPOCH500_CLEAN", flush=True)
def barrier(original):
    return watcher.make_cleanup_barrier(
        original, pause=signal.pause, announce=announce)
def guard_engine(engine, task, expected):
    original=engine.run
    def guarded(*args,**kwargs):
        if watcher._probe_task_state(task) not in expected:
            raise RuntimeError("run state changed after the runner lock was acquired")
        return original(*args,**kwargs)
    engine.run=guarded
def forbid_completed_finalization(*args,**kwargs):
    raise RuntimeError("preexisting completed transaction is forbidden in Stage 1")

if kind == "psbfr":
    import train_irstd_model_design_screen_v1 as runner
    args=runner.parse_args([
        "--dataset-root","/home/ly/SCTransNet_main/datasets",
        "--split-root","/home/ly/EviSIRST_main/splits/v2",
        "--variant","psbfr_v1","--dataset","IRSTD-1K",
        "--target-mode","binary","--architecture-seed","42",
        "--run-seed",str(seed),"--device","cuda:0","--epochs","1000",
        "--warmup-epochs","10","--allow-sample-level-fallback",
        *(["--resume"] if resume else []),
    ])
    runner.r1._validate_resume_candidates=barrier(
        runner.r1._validate_resume_candidates)
    guard_engine(runner.r1,matches[0],expected_states)
    runner.hf_transaction._finalize_completed=forbid_completed_finalization
    runner.run(args)
elif kind == "cp_hf_s2":
    import train_irstd_cp_hf_s2_legacy_screen_v1 as runner
    import train_irstd_model_design_screen_v1 as shared
    args=runner.parse_args([
        "--dataset-root","/home/ly/SCTransNet_main/datasets",
        "--run-seed",str(seed),"--device","cuda:0","--epochs","1000",
        "--warmup-epochs","10",*(["--resume"] if resume else []),
    ])
    shared.r1._validate_resume_candidates=barrier(
        shared.r1._validate_resume_candidates)
    guard_engine(shared.r1,matches[0],expected_states)
    shared.hf_transaction._finalize_completed=forbid_completed_finalization
    runner.run(args)
elif kind == "clean":
    if not resume:
        raise RuntimeError("clean R1 is resume-only")
    import train_validation_selected as runner
    args=runner.parse_args([
        "--dataset","IRSTD-1K",
        "--dataset-root","/home/ly/SCTransNet_main/datasets",
        "--split-root","/home/ly/EviSIRST_main/splits/v2",
        "--output-root","/home/ly/EviSIRST_main/runs/validation_selected",
        "--target-mode","binary","--architecture-seed","42",
        "--run-seed",str(seed),"--device","cuda:0","--epochs","1000",
        "--batch-size","16","--workers","0","--base-lr","0.001",
        "--min-lr","0.00001","--warmup-epochs","10",
        "--val-interval","1","--allow-sample-level-fallback","--resume",
    ])
    runner._validate_resume_candidates=barrier(
        runner._validate_resume_candidates)
    runner.run(args)
else:
    raise RuntimeError("unknown fixed A/B task kind")
raise RuntimeError("formal worker returned without controlled epoch-500 stop")
'''


def worker_argv(task: Task, *, resume: bool) -> tuple[str, ...]:
    if task.name not in STAGE1_TASK_NAMES:
        raise ABModelScreenWatcherError(
            "task is locked until a future fixed S1 decision amendment"
        )
    if not resume and not task.fresh_allowed:
        raise ABModelScreenWatcherError("clean R1 is a resume-only task")
    return (
        os.fspath(PYTHON_BIN),
        "-c",
        _FORMAL_WORKER_PROGRAM,
        task.kind,
        str(task.run_seed),
        "resume" if resume else "fresh",
    )


def timed_worker_argv(task: Task, *, resume: bool) -> tuple[str, ...]:
    return timed_argv(worker_argv(task, resume=resume))


_STATE_PROBE_PROGRAM = r'''
import hashlib, json, math, os, stat, sys
from contextlib import ExitStack
from pathlib import Path

import torch

PREFIX="EVISIRST_AB_MODEL_SCREEN_PROBE:"
kind=sys.argv[1]
seed=int(sys.argv[2])
minimum_epoch=int(sys.argv[3])

def emit(value): print(PREFIX+value, flush=True)
def fail(message): raise RuntimeError(message)
def regular(path):
    st=path.lstat()
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        fail("not a regular artifact")
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()
def finite(value):
    if isinstance(value,float) and not math.isfinite(value): fail("nonfinite")
    if isinstance(value,dict):
        for key,nested in value.items():
            if key in {"lockbox_accessed","test_split_accessed","public_test_allowed","public_test_supported","test_index_opened"} and nested is not False:
                fail("forbidden disclosure")
            finite(nested)
    elif isinstance(value,list):
        for nested in value: finite(nested)
def unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result: fail("duplicate JSON key")
        result[key]=value
    return result
def safe_run_chain(path):
    root=Path("/home/ly/EviSIRST_main/runs")
    try: relative=path.relative_to(root)
    except ValueError: fail("run directory escaped runs")
    current=root
    for component in relative.parts:
        current=current/component
        if current.is_symlink(): fail("run directory chain contains a symlink")
        if current.exists() and not current.is_dir():
            fail("run directory chain contains a non-directory")
        if not current.exists(): break

with ExitStack() as stack:
    if kind == "psbfr":
        import train_irstd_model_design_screen_v1 as front
        args=front.parse_args([
            "--dataset-root","/home/ly/SCTransNet_main/datasets",
            "--split-root","/home/ly/EviSIRST_main/splits/v2",
            "--variant","psbfr_v1","--dataset","IRSTD-1K",
            "--target-mode","binary","--architecture-seed","42",
            "--run-seed",str(seed),"--device","cuda:0","--epochs","1000",
            "--warmup-epochs","10","--allow-sample-level-fallback","--resume",
        ])
        stack.enter_context(front._hf_adapter(args))
        engine=front.r1
        transaction=front.hf_transaction
        paths=front.resolve_run_paths(args)
        identity=transaction._current_identity_for_args(args)
        model,_=front._variant_initialize_evisirst(
            "IRSTD-1K",seed=42,training=True)
        validate_state=front._validate_state_dict
        selector=front.zero_selection
        zero_margin=True
        training_schema=front.TRAINING_SCHEMA
        candidate_schema=front.CANDIDATE_SCHEMA
        history_schema=front.HISTORY_SCHEMA
        permitted_lock=".model_design_screen_v1.lock"
    elif kind == "cp_hf_s2":
        import train_irstd_cp_hf_s2_legacy_screen_v1 as front
        args=front.parse_args([
            "--dataset-root","/home/ly/SCTransNet_main/datasets",
            "--run-seed",str(seed),"--device","cuda:0","--epochs","1000",
            "--warmup-epochs","10","--resume",
        ])
        shared=stack.enter_context(front._screen_transaction_adapter(args))
        stack.enter_context(shared._hf_adapter(args))
        engine=shared.r1
        transaction=shared.hf_transaction
        paths=shared.resolve_run_paths(args)
        identity=transaction._current_identity_for_args(args)
        model,_=shared._variant_initialize_evisirst(
            "IRSTD-1K",seed=42,training=True)
        validate_state=shared._validate_state_dict
        selector=shared.zero_selection
        zero_margin=True
        training_schema=front.TRAINING_SCHEMA
        candidate_schema=front.CANDIDATE_SCHEMA
        history_schema=front.HISTORY_SCHEMA
        permitted_lock=".model_design_screen_v1.lock"
    elif kind == "clean":
        import train_validation_selected as front
        args=front.parse_args([
            "--dataset","IRSTD-1K",
            "--dataset-root","/home/ly/SCTransNet_main/datasets",
            "--split-root","/home/ly/EviSIRST_main/splits/v2",
            "--output-root","/home/ly/EviSIRST_main/runs/validation_selected",
            "--target-mode","binary","--architecture-seed","42",
            "--run-seed",str(seed),"--device","cuda:0","--epochs","1000",
            "--batch-size","16","--workers","0","--base-lr","0.001",
            "--min-lr","0.00001","--warmup-epochs","10",
            "--val-interval","1","--allow-sample-level-fallback","--resume",
        ])
        engine=front
        paths=front.resolve_run_paths(args)
        full_train,full_val,grouping=front.build_datasets(args)
        identity=front._run_identity(
            args,full_train.contract,grouping,
            train_count=len(full_train),val_count=len(full_val),smoke=False)
        model,_=front.initialize_evisirst("IRSTD-1K",seed=42,training=True)
        validate_state=front._validate_state_dict
        selector=front.selection
        zero_margin=False
        training_schema=front.TRAINING_SCHEMA
        candidate_schema=front.CANDIDATE_SCHEMA
        history_schema=front.HISTORY_SCHEMA
        permitted_lock=None
    else:
        fail("unknown task kind")

    run=paths["run_dir"]
    safe_run_chain(run)
    if run.is_symlink(): emit("CONFLICT"); raise SystemExit(0)
    if not run.exists(): emit("FRESH"); raise SystemExit(0)
    if not run.is_dir(): emit("CONFLICT"); raise SystemExit(0)
    latest=paths["latest"]
    history_path=paths["history"]
    candidate_dir=paths["candidate_dir"]
    forbidden=[]
    for name in ("summary","final","best_mIoU_final","best_Pd_final"):
        path=paths.get(name)
        if path is not None: forbidden.append(path)
    if any(path.exists() or path.is_symlink() for path in forbidden):
        emit("CONFLICT"); raise SystemExit(0)
    permitted={latest.name,history_path.name,candidate_dir.name}
    if permitted_lock is not None: permitted.add(permitted_lock)
    actual={path.name for path in run.iterdir()}
    if not latest.exists():
        if actual <= ({permitted_lock} if permitted_lock is not None else set()):
            emit("FRESH")
        else:
            emit("CONFLICT")
        raise SystemExit(0)
    if actual - permitted:
        emit("CONFLICT"); raise SystemExit(0)
    regular(latest); regular(history_path)
    latest_value=torch.load(latest,map_location="cpu",weights_only=True)
    finite(latest_value)
    if (not isinstance(latest_value,dict)
        or latest_value.get("schema") != training_schema
        or latest_value.get("run_identity") != identity
        or latest_value.get("test_split_accessed") is not False):
        fail("latest identity differs")
    epoch=latest_value.get("epoch")
    if isinstance(epoch,bool) or not isinstance(epoch,int) or not minimum_epoch <= epoch <= 500:
        fail("epoch outside fixed screen interval")
    validate_state(latest_value.get("state_dict"),model.state_dict())
    optimizer=torch.optim.Adam(model.parameters(),lr=args.base_lr)
    engine._validate_and_load_adam_optimizer_state(
        optimizer_state=latest_value.get("optimizer"),model=model,
        optimizer=optimizer,identity=identity,completed_epoch=epoch,total_epochs=1000)
    train,val=engine._validate_resume_history(
        completed_epoch=epoch,interval=1,
        training_history=latest_value.get("training_history"),
        validation_history=latest_value.get("validation_history"))
    if (len(train)!=epoch or len(val)!=epoch or train[-1]["epoch"]!=epoch
        or val[-1]["epoch"]!=epoch):
        fail("history endpoint differs")
    artifacts=latest_value.get("candidate_artifacts")
    frontier=(tuple(selector.retention_frontier_epochs(val,margin=None))
              if zero_margin else tuple(selector.retention_frontier_epochs(val)))
    if not isinstance(artifacts,dict) or tuple(sorted(artifacts)) != frontier:
        fail("candidate frontier differs")
    if candidate_dir.is_symlink() or not candidate_dir.is_dir():
        fail("candidate directory differs")
    names=set(); by_epoch={record["epoch"]:record for record in val}
    for candidate_epoch in frontier:
        metadata=artifacts[candidate_epoch]
        name="epoch_%04d.pth.tar"%candidate_epoch
        path=candidate_dir/name; names.add(name); regular(path)
        if metadata != {"relative_path":"candidates/"+name,"file_sha256":sha(path)}:
            fail("candidate metadata differs")
        payload=torch.load(path,map_location="cpu",weights_only=True); finite(payload)
        if (payload.get("schema") != candidate_schema
            or payload.get("run_identity") != identity
            or payload.get("epoch") != candidate_epoch
            or payload.get("validation_record") != by_epoch[candidate_epoch]
            or payload.get("test_split_accessed") is not False):
            fail("candidate identity differs")
        validate_state(payload.get("state_dict"),model.state_dict())
    if {path.name for path in candidate_dir.iterdir()} != names:
        fail("candidate cleanup is incomplete")
    raw=json.loads(
        history_path.read_text(encoding="utf-8"),
        parse_constant=lambda token:fail("nonfinite JSON constant"),
        object_pairs_hook=unique_object)
    finite(raw)
    if (raw.get("schema") != history_schema
        or raw.get("run_identity") != identity
        or raw.get("training_history") != train
        or raw.get("validation_history") != val
        or raw.get("candidate_artifacts") != {str(key):value for key,value in artifacts.items()}
        or raw.get("test_split_accessed") is not False):
        fail("history JSON differs")
    emit("READY" if epoch == 500 else "RESUME")
'''


def _probe_task_state(task: Task) -> str:
    try:
        _assert_safe_task_run_chain(task)
    except (ABModelScreenWatcherError, OSError):
        return "conflict"
    if not frozen_sources_are_valid():
        return "conflict"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    try:
        completed = subprocess.run(
            [
                os.fspath(PYTHON_BIN),
                "-c",
                _STATE_PROBE_PROGRAM,
                task.kind,
                str(task.run_seed),
                str(task.minimum_resume_epoch),
            ],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "conflict"
    tokens = [
        line[len(_PROBE_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(_PROBE_PREFIX)
    ]
    if completed.returncode != 0 or len(tokens) != 1:
        return "conflict"
    value = tokens[0].lower()
    return value if value in {"fresh", "resume", "ready", "conflict"} else "conflict"


_STAGE1_CONTROL_PROBE_PROGRAM = r'''
import sys
import run_irstd_model_design_epoch500_gate_v1 as gate

PREFIX="EVISIRST_AB_MODEL_SCREEN_CONTROL_PROBE:"
evidence=gate._read_arm(seed=42,variant=None)
selection=evidence.get("fresh_zero_margin_selection")
roles=selection.get("roles") if isinstance(selection,dict) else None
primary=roles.get("best_mIoU") if isinstance(roles,dict) else None
selected=primary.get("selected") if isinstance(primary,dict) else None
if (evidence.get("role") != "clean_control"
    or evidence.get("run_seed") != 42
    or evidence.get("configured_total_epochs") != 1000
    or evidence.get("screen_prefix_epochs") != 500
    or not isinstance(evidence.get("committed_epoch"),int)
    or evidence["committed_epoch"] < 500
    or evidence.get("test_split_accessed") is not False
    or not isinstance(selected,dict)
    or not 1 <= selected.get("epoch",0) <= 500):
    raise RuntimeError("clean Seed-42 first-500 control differs")
print(PREFIX+"READY",flush=True)
'''


def stage1_clean_control_is_valid() -> bool:
    try:
        assert_frozen_sources()
        assert_frozen_stage1_clean_control_artifacts()
    except (ABModelScreenWatcherError, OSError):
        return False
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    try:
        completed = subprocess.run(
            [os.fspath(PYTHON_BIN), "-c", _STAGE1_CONTROL_PROBE_PROGRAM],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    tokens = [
        line[len(_CONTROL_PROBE_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(_CONTROL_PROBE_PREFIX)
    ]
    return completed.returncode == 0 and tokens == ["READY"]


def _read_cmdline(pid_dir: Path) -> tuple[str, ...] | None:
    try:
        raw = (pid_dir / "cmdline").read_bytes()
    except (OSError, PermissionError):
        return None
    tokens = raw.rstrip(b"\0").split(b"\0") if raw else []
    try:
        return tuple(token.decode("utf-8", errors="strict") for token in tokens)
    except UnicodeDecodeError:
        return None


def exact_argv_processes(
    expected: Sequence[str],
    *,
    proc_root: Path = PROC_ROOT,
    exclude_pid: int | None = None,
) -> tuple[int, ...]:
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as exc:
        raise ABModelScreenWatcherError("cannot enumerate /proc") from exc
    wanted = tuple(expected)
    found = []
    for entry in entries:
        if not entry.name.isascii() or not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        if pid != exclude_pid and _read_cmdline(entry) == wanted:
            found.append(pid)
    return tuple(sorted(found))


def task_processes(
    task: Task,
    *,
    resume: bool,
    proc_root: Path = PROC_ROOT,
) -> tuple[int, ...]:
    try:
        trainer = formal_training_argv(task, resume=resume)
    except ABModelScreenWatcherError:
        return ()
    candidates: list[tuple[str, ...]] = [trainer, timed_argv(trainer)]
    try:
        worker = worker_argv(task, resume=resume)
    except ABModelScreenWatcherError:
        pass
    else:
        candidates.extend((worker, timed_argv(worker)))
    found: set[int] = set()
    for argv in candidates:
        found.update(exact_argv_processes(argv, proc_root=proc_root))
    return tuple(sorted(found))


def _completion_path(task: Task) -> Path:
    return COMPLETION_ROOT / f"{task.name}.json"


def _completion_payload(task: Task) -> dict[str, object]:
    latest = task.run_dir / "last_training_state.pth.tar"
    history = task.run_dir / "validation_history.json"
    return {
        "schema": COMPLETION_SCHEMA,
        "task": task.name,
        "kind": task.kind,
        "variant": task.variant,
        "run_seed": task.run_seed,
        "minimum_resume_epoch": task.minimum_resume_epoch,
        "configured_total_epochs": FINAL_CONFIGURED_EPOCH,
        "stopped_after_epoch": STOP_EPOCH,
        "latest_sha256": _sha256_file(latest),
        "history_sha256": _sha256_file(history),
        "task_manifest_sha256": _canonical_sha256(task_manifest()),
        "source_allowlist_sha256": frozen_source_allowlist_sha256(),
        "watcher_source_sha256": _watcher_source_sha256(),
        "authorization_sha256": FORMAL_AUTHORIZATION_SHA256,
        "candidate_cleanup_validated": True,
        "lockbox_accessed": False,
        "public_test_allowed": False,
        "test_split_accessed": False,
    }


def _completion_is_valid(task: Task) -> bool:
    path = _completion_path(task)
    if not path.exists():
        return False
    try:
        payload = _strict_json(path)
        expected = _completion_payload(task)
    except (ABModelScreenWatcherError, OSError):
        return False
    return isinstance(payload, dict) and payload == expected


def task_state(task: Task) -> str:
    if (
        task.name == "clean_r1_s104728269_to500"
        and not stage2_clean_resume_is_frozen()
    ):
        return "conflict"
    running = task_processes(task, resume=False) or task_processes(task, resume=True)
    probed = _probe_task_state(task)
    if running:
        return "conflict"
    if probed == "ready":
        marker = _completion_path(task)
        if marker.exists() or marker.is_symlink():
            return "complete" if _completion_is_valid(task) else "conflict"
        # A crash/reboot after the atomic epoch-500 commit but before the
        # watcher ledger is recoverable.  Resume re-enters the same validated
        # candidate-cleanup barrier and emits a fresh post-cleanup marker.
        return "resume"
    if probed == "fresh":
        return "fresh" if task.fresh_allowed else "conflict"
    if probed == "resume":
        return "resume"
    return "conflict"


def _parse_gpu_rows(text: str) -> dict[str, GPU]:
    result: dict[str, GPU] = {}
    indexes: set[int] = set()
    buses: set[str] = set()
    for raw_line in text.splitlines():
        fields = tuple(field.strip() for field in raw_line.split(","))
        if len(fields) != 4:
            raise ABModelScreenWatcherError("unexpected nvidia-smi GPU row")
        try:
            index, memory = int(fields[0]), int(fields[3])
        except ValueError as exc:
            raise ABModelScreenWatcherError("non-integer GPU field") from exc
        uuid, bus = fields[1], fields[2].lower()
        if (
            not _UUID_RE.fullmatch(uuid)
            or uuid in result
            or index in indexes
            or bus in buses
            or index < 0
            or memory < 0
            or not bus
        ):
            raise ABModelScreenWatcherError("ambiguous NVIDIA GPU identity")
        result[uuid] = GPU(index, uuid, bus, memory)
        indexes.add(index)
        buses.add(bus)
    if not result:
        raise ABModelScreenWatcherError("nvidia-smi returned no GPUs")
    return result


def _parse_compute_uuids(text: str) -> set[str]:
    occupied: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("No running processes"):
            continue
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 2 or not _UUID_RE.fullmatch(fields[0]):
            raise ABModelScreenWatcherError("unexpected compute-app row")
        try:
            int(fields[1])
        except ValueError as exc:
            raise ABModelScreenWatcherError("non-integer compute PID") from exc
        occupied.add(fields[0])
    return occupied


def sample_idle_gpus() -> dict[str, GPU]:
    gpu_rows = subprocess.run(
        [
            os.fspath(NVIDIA_SMI),
            "--query-gpu=index,uuid,pci.bus_id,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    apps = subprocess.run(
        [
            os.fspath(NVIDIA_SMI),
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if gpu_rows.returncode != 0 or apps.returncode != 0:
        raise ABModelScreenWatcherError("nvidia-smi query failed")
    gpus = _parse_gpu_rows(gpu_rows.stdout)
    occupied = _parse_compute_uuids(apps.stdout)
    return {
        uuid: gpu
        for uuid, gpu in gpus.items()
        if uuid not in occupied and gpu.memory_used_mib <= GPU_MEMORY_LIMIT_MIB
    }


def _stable_intersection(samples: Sequence[Mapping[str, GPU]]) -> dict[str, GPU]:
    if not samples:
        return {}
    common = set(samples[0])
    for sample in samples[1:]:
        common.intersection_update(sample)
    result = {}
    for uuid in common:
        identities = {(sample[uuid].index, sample[uuid].bus_id) for sample in samples}
        if len(identities) == 1:
            result[uuid] = samples[-1][uuid]
    return result


def _open_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ABModelScreenWatcherError(f"lock is a symlink: {path}")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags, 0o600)


def _try_lock(path: Path) -> int | None:
    descriptor = _open_lock(path)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        return None
    return descriptor


def claim_idle_gpus(
    *, sleep: Callable[[float], None] = time.sleep
) -> list[tuple[GPU, int]]:
    first = sample_idle_gpus()
    sleep(GPU_CONFIRM_SECONDS)
    second = sample_idle_gpus()
    candidates = _stable_intersection((first, second))
    locked: list[tuple[GPU, int]] = []
    try:
        for gpu in sorted(candidates.values(), key=lambda item: item.index):
            descriptor = _try_lock(GPU_LOCK_ROOT / f"{gpu.uuid}.lock")
            if descriptor is not None:
                locked.append((gpu, descriptor))
        if not locked:
            return []
        third = sample_idle_gpus()
        sleep(GPU_CONFIRM_SECONDS)
        fourth = sample_idle_gpus()
        confirmed = _stable_intersection((third, fourth))
        accepted = []
        for gpu, descriptor in locked:
            observed = confirmed.get(gpu.uuid)
            if observed and (observed.index, observed.bus_id) == (
                gpu.index,
                gpu.bus_id,
            ):
                accepted.append((observed, descriptor))
            else:
                os.close(descriptor)
        return accepted
    except BaseException:
        for _gpu, descriptor in locked:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _scrubbed_gpu_env(gpu: GPU) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu.uuid
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    return env


def _open_log(path: Path) -> object:
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ABModelScreenWatcherError("task log is not a regular file")
        return os.fdopen(descriptor, "ab", buffering=0)
    except BaseException:
        os.close(descriptor)
        raise


def launch_task(
    task: Task,
    *,
    resume: bool,
    gpu: GPU,
    gpu_lock_fd: int,
) -> RunningJob | None:
    if task.name not in STAGE1_TASK_NAMES:
        os.close(gpu_lock_fd)
        raise ABModelScreenWatcherError(
            "Stage-2 task launch is not implemented by this watcher"
        )
    try:
        authorized = authorization_is_valid()
    except BaseException:
        os.close(gpu_lock_fd)
        raise
    if not authorized:
        os.close(gpu_lock_fd)
        return None
    try:
        task_fd = _try_lock(TASK_LOCK_ROOT / f"{task.name}.lock")
    except BaseException:
        os.close(gpu_lock_fd)
        raise
    if task_fd is None:
        os.close(gpu_lock_fd)
        return None
    try:
        argv = worker_argv(task, resume=resume)
        if task_processes(task, resume=False) or task_processes(task, resume=True):
            os.close(task_fd)
            os.close(gpu_lock_fd)
            return None
        final_idle = sample_idle_gpus()
        observed = final_idle.get(gpu.uuid)
        if observed is None or (observed.index, observed.bus_id) != (
            gpu.index,
            gpu.bus_id,
        ):
            os.close(task_fd)
            os.close(gpu_lock_fd)
            return None
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        log_path = LOG_ROOT / f"{task.name}.log"
        if log_path.is_symlink() or (log_path.exists() and not log_path.is_file()):
            raise ABModelScreenWatcherError("task log path is unsafe")
        log_handle = _open_log(log_path)
        log_start_offset = log_handle.tell()
        try:
            process = subprocess.Popen(
                list(timed_argv(argv)),
                cwd=PROJECT_ROOT,
                env=_scrubbed_gpu_env(gpu),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                pass_fds=(gpu_lock_fd, task_fd),
                start_new_session=True,
            )
        except BaseException:
            log_handle.close()
            raise
    except BaseException:
        try:
            os.close(task_fd)
        finally:
            os.close(gpu_lock_fd)
        raise
    return RunningJob(
        task,
        process,
        gpu,
        gpu_lock_fd,
        task_fd,
        log_handle,
        log_path,
        log_start_offset,
    )


def _release_job(job: RunningJob) -> None:
    first_error: BaseException | None = None
    try:
        job.log_handle.close()
    except BaseException as exc:
        first_error = exc
    for name in ("task_lock_fd", "gpu_lock_fd"):
        descriptor = getattr(job, name)
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            finally:
                setattr(job, name, -1)
    if first_error is not None:
        raise first_error


def _abort_job(job: RunningJob) -> None:
    """Best-effort containment for every worker left by supervisor exit."""

    try:
        if job.process.poll() is None:
            try:
                os.killpg(job.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                job.process.wait(timeout=TERM_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(job.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                job.process.wait(timeout=TERM_TIMEOUT_SECONDS)
    except BaseException as exc:
        _log(f"worker containment failed task={job.task.name}: {exc!r}")
    finally:
        try:
            _release_job(job)
        except BaseException as exc:
            _log(f"worker resource release failed task={job.task.name}: {exc!r}")


def _job_announced_post_cleanup(job: RunningJob) -> bool:
    marker = b"EVISIRST_AB_MODEL_SCREEN_WORKER:EPOCH500_CLEAN\n"
    try:
        if job.log_path.is_symlink() or not job.log_path.is_file():
            return False
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(job.log_path, flags)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            return False
        with os.fdopen(descriptor, "rb") as handle:
            handle.seek(job.log_start_offset)
            new_output = handle.read()
    except OSError:
        return False
    return marker in new_output


def _atomic_write_completion(task: Task) -> Path:
    COMPLETION_ROOT.mkdir(parents=True, exist_ok=True)
    payload = _completion_payload(task)
    path = _completion_path(task)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise ABModelScreenWatcherError("completion marker no-clobber conflict")
        return path
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=COMPLETION_ROOT
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except OSError as exc:
            if (
                exc.errno == errno.EEXIST
                and path.is_file()
                and not path.is_symlink()
                and path.read_bytes() == encoded
            ):
                return path
            raise ABModelScreenWatcherError(
                "completion marker no-clobber conflict"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def stop_ready_job(job: RunningJob) -> bool:
    """Validate epoch 500, TERM its process group, then validate once more."""

    if (
        job.process.poll() is not None
        or not _job_announced_post_cleanup(job)
        or _probe_task_state(job.task) != "ready"
    ):
        return False
    try:
        os.killpg(job.process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    try:
        job.process.wait(timeout=TERM_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(job.process.pid, signal.SIGKILL)
        job.process.wait(timeout=TERM_TIMEOUT_SECONDS)
        raise ABModelScreenWatcherError("epoch-500 worker ignored SIGTERM")
    if _probe_task_state(job.task) != "ready":
        raise ABModelScreenWatcherError("epoch-500 transaction changed after TERM")
    _atomic_write_completion(job.task)
    if not _completion_is_valid(job.task):
        raise ABModelScreenWatcherError("epoch-500 completion verification failed")
    return True


def _supervise_unprotected(
    *,
    running: dict[str, RunningJob],
    sleep: Callable[[float], None],
) -> int:
    if not authorization_is_valid():
        raise ABModelScreenWatcherError("formal launch authorization is not frozen")
    if not stage1_clean_control_is_valid():
        raise ABModelScreenWatcherError(
            "clean Seed-42 first-500 control validation failed"
        )
    while True:
        for name, job in tuple(running.items()):
            if (
                _job_announced_post_cleanup(job)
                and _probe_task_state(job.task) == "ready"
            ):
                if not stop_ready_job(job):
                    raise ABModelScreenWatcherError(
                        "controlled stop precondition failed"
                    )
                _release_job(job)
                del running[name]
                continue
            returncode = job.process.poll()
            if returncode is not None:
                _release_job(job)
                del running[name]
                raise ABModelScreenWatcherError(
                    f"{name} exited before a clean epoch-500 commit: rc={returncode}"
                )

        states = {
            task.name: (
                "running" if task.name in running else task_state(task)
            )
            for task in TASKS
        }
        for name, state_value in states.items():
            _log(f"task={name} state={state_value}")
        for task in TASKS:
            if task.name not in STAGE2_LOCKED_TASK_NAMES:
                continue
            expected_locked_state = "resume" if task.kind == "clean" else "fresh"
            if states[task.name] != expected_locked_state:
                raise ABModelScreenWatcherError(
                    f"locked Stage-2 task changed state: {task.name}"
                )
        stage1_states = {name: states[name] for name in STAGE1_TASK_NAMES}
        if all(value == "complete" for value in stage1_states.values()):
            _log(
                "Stage 1 complete; Stage 2 remains locked pending fixed "
                "route-specific S1 decision artifacts"
            )
            return 0
        if any(value == "conflict" for value in states.values()):
            raise ABModelScreenWatcherError("A/B task queue contains a conflict")
        pending = [
            task
            for task in TASKS
            if task.name in STAGE1_TASK_NAMES
            and states[task.name] in {"fresh", "resume"}
            and task.name not in running
        ]
        if not pending:
            sleep(POLL_SECONDS)
            continue
        available = claim_idle_gpus(sleep=sleep)
        for gpu, gpu_fd in available:
            if not pending:
                os.close(gpu_fd)
                continue
            task = pending.pop(0)
            job = launch_task(
                task,
                resume=states[task.name] == "resume",
                gpu=gpu,
                gpu_lock_fd=gpu_fd,
            )
            if job is None:
                continue
            running[task.name] = job
            _log(
                f"launched {task.name} pid={job.process.pid} "
                f"gpu={gpu.uuid} index={gpu.index} bus={gpu.bus_id}"
            )
        sleep(POLL_SECONDS)


def supervise(*, sleep: Callable[[float], None] = time.sleep) -> int:
    running: dict[str, RunningJob] = {}
    try:
        return _supervise_unprotected(running=running, sleep=sleep)
    finally:
        for job in tuple(running.values()):
            _abort_job(job)
        running.clear()


def _require_safe_directory_chain(path: Path) -> None:
    runs = PROJECT_ROOT / "runs"
    if runs.is_symlink() or (runs.exists() and not runs.is_dir()):
        raise ABModelScreenWatcherError("repository runs root is unsafe")
    try:
        relative = path.relative_to(runs)
    except ValueError as exc:
        raise ABModelScreenWatcherError("watcher state escaped runs") from exc
    current = runs
    for component in relative.parts:
        current /= component
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise ABModelScreenWatcherError(f"unsafe watcher path: {current}")


def readonly_preflight() -> dict[str, object]:
    """Inspect dependencies, exact processes, task state, and GPUs without writes."""

    python_target = PYTHON_BIN.resolve(strict=True)
    if (
        python_target != EXPECTED_PYTHON
        or not python_target.is_file()
        or not os.access(python_target, os.X_OK)
    ):
        raise ABModelScreenWatcherError("fixed Python target differs")
    for required in (
        CLEAN_TRAINER,
        PSBFR_TRAINER,
        CP_HF_S2_TRAINER,
        STAGE1_CONTROL_READER,
        WATCHER_SOURCE,
        TIME_BIN,
        NVIDIA_SMI,
    ):
        if required.is_symlink() or not required.is_file():
            raise ABModelScreenWatcherError(f"fixed dependency differs: {required}")
    assert_frozen_sources()
    assert_frozen_stage1_clean_control_artifacts()
    assert_frozen_stage2_clean_resume_artifacts()
    for path in (WATCH_ROOT, TASK_LOCK_ROOT, LOG_ROOT, COMPLETION_ROOT, GPU_LOCK_ROOT):
        _require_safe_directory_chain(path)
    states = {task.name: task_state(task) for task in TASKS}
    clean_control_valid = stage1_clean_control_is_valid()
    idle = sample_idle_gpus()
    stage2_lock_intact = all(
        states[task.name] == ("resume" if task.kind == "clean" else "fresh")
        for task in TASKS
        if task.name in STAGE2_LOCKED_TASK_NAMES
    )
    return {
        "schema": "evisirst_irstd_ab_model_screen_readonly_preflight/v1",
        "formal_launch_authorized": authorization_is_valid(),
        "authorization_sha256_frozen": FORMAL_AUTHORIZATION_SHA256 is not None,
        "task_manifest_sha256": _canonical_sha256(task_manifest()),
        "watcher_source_sha256": _watcher_source_sha256(),
        "source_allowlist_sha256": frozen_source_allowlist_sha256(),
        "task_states": states,
        "launcher_stage": "S1_seed42_only",
        "stage1_task_names": sorted(STAGE1_TASK_NAMES),
        "stage2_locked_task_names": sorted(STAGE2_LOCKED_TASK_NAMES),
        "stage2_lock_intact": stage2_lock_intact,
        "s2_unlock_implemented": False,
        "stage1_clean_control_valid": clean_control_valid,
        "stage2_clean_resume_frozen": stage2_clean_resume_is_frozen(),
        "idle_gpu_uuids": sorted(idle),
        "candidate_decisions_are_independent": True,
        "candidate_ranking_or_winner_selection": False,
        "lockbox_accessed": False,
        "public_test_supported": False,
        "test_split_accessed": False,
        "writes_performed": False,
        "workers_launched": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--manifest", action="store_true")
    modes.add_argument("--formal", action="store_true")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.preflight:
        print(json.dumps(readonly_preflight(), sort_keys=True, indent=2))
        return 0
    if args.manifest:
        print(json.dumps(task_manifest(), sort_keys=True, indent=2))
        return 0
    if not authorization_is_valid():
        raise ABModelScreenWatcherError(
            "formal mode disabled: freeze authorization file and SHA-256 first"
        )
    for path in (WATCH_ROOT, TASK_LOCK_ROOT, LOG_ROOT, COMPLETION_ROOT, GPU_LOCK_ROOT):
        _require_safe_directory_chain(path)
    singleton = _try_lock(WATCHER_LOCK)
    if singleton is None:
        _log("another A/B watcher owns the singleton lock")
        return 0
    try:
        return supervise()
    finally:
        os.close(singleton)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ABModelScreenWatcherError",
    "FROZEN_SOURCE_SHA256",
    "GPU",
    "STAGE1_TASK_NAMES",
    "STAGE2_LOCKED_TASK_NAMES",
    "TASKS",
    "Task",
    "assert_frozen_stage1_clean_control_artifacts",
    "assert_frozen_sources",
    "authorization_is_valid",
    "claim_idle_gpus",
    "exact_argv_processes",
    "formal_training_argv",
    "frozen_source_allowlist_sha256",
    "main",
    "make_cleanup_barrier",
    "readonly_preflight",
    "sample_idle_gpus",
    "stage1_clean_control_is_valid",
    "stop_ready_job",
    "supervise",
    "task_manifest",
    "task_processes",
    "task_state",
    "timed_argv",
    "timed_worker_argv",
    "worker_argv",
]
