#!/usr/bin/env python3
"""Fail-closed two-arm watcher for the formal IRSTD-1K DCS-PG run.

This watcher authorizes exactly ``dcspg_original`` and ``dcspg_farbg``.
It pins each arm to one full GPU UUID/PCI identity, passes three locked file
descriptors to the capability-gated adapter, and contains every worker process
group on SIGINT/SIGTERM.  The authorization literal and source allow-list are
intentionally unsealed in this candidate; consequently formal execution is
disabled until a separate audit freezes them.

The quarantined pre-audit output is deliberately never inspected or reused.
Only the new ``test_selected_v1`` output root is eligible, and its initial
formal launch requires that root to be completely absent (including symlinks).
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import importlib
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
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Callable, Mapping, Sequence


PROJECT_ROOT = Path("/home/ly/EviSIRST_main")
# Direct execution sets ``sys.path[0]`` to ``tools/``.  Completion validation
# imports the frozen top-level runner by module name, so make the canonical
# project root available before that lazy import.  The path is fixed rather
# than derived from the process working directory.
if os.fspath(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(PROJECT_ROOT))
PYTHON_BIN = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
EXPECTED_PYTHON = Path("/usr/bin/python3.12")
TIME_BIN = Path("/usr/bin/time")
NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
PROC_ROOT = Path("/proc")

ADAPTER_SOURCE = PROJECT_ROOT / "run_irstd_dcspg_test_selected_v1.py"
RUNNER_SOURCE = PROJECT_ROOT / "train_irstd_dcspg_ablation_test_selected_v1.py"
ARCHITECTURE_SOURCE = PROJECT_ROOT / "experiments/irstd_cp_hf_s2_dcspg_v1.py"
ARCHITECTURE_TEST_SOURCE = PROJECT_ROOT / "tests/test_irstd_cp_hf_s2_dcspg_v1.py"
ADAPTER_TEST_SOURCE = PROJECT_ROOT / "tests/test_irstd_dcspg_test_selected_adapter.py"
RUNNER_TEST_SOURCE = (
    PROJECT_ROOT / "tests/test_train_irstd_dcspg_ablation_test_selected_v1.py"
)
PROTOCOL_SOURCE = (
    PROJECT_ROOT / "experiments/IRSTD_DCSPG_ABLATION_TEST_SELECTED_V1_PROTOCOL.md"
)
RULES_SOURCE = (
    PROJECT_ROOT / "experiments/irstd_dcspg_ablation_test_selected_v1_rules.json"
)
WATCHER_SOURCE = PROJECT_ROOT / "tools/run_irstd_dcspg_test_selected_when_idle.py"
WATCHER_TEST_SOURCE = PROJECT_ROOT / "tests/test_irstd_dcspg_test_selected_watcher.py"

FORMAL_OUTPUT_ROOT = (
    PROJECT_ROOT / "runs/irstd_model_design/dcspg_ablation_v1/test_selected_v1"
)
QUARANTINED_OUTPUT_ROOT = PROJECT_ROOT / (
    "runs/irstd_model_design/dcspg_ablation_v1/"
    "test_selected_v1_invalid_pre_audit_20260819T043550"
)
WATCH_ROOT = (
    PROJECT_ROOT
    / "runs/irstd_model_design/dcspg_ablation_v1/test_selected_v1_watcher_v1"
)
WATCHER_LOCK = WATCH_ROOT / "watcher.lock"
TASK_LOCK_ROOT = WATCH_ROOT / "task_locks"
LOG_ROOT = WATCH_ROOT / "logs"
COMPLETION_ROOT = WATCH_ROOT / "completion"
INITIALIZATION_PATH = WATCH_ROOT / "formal_initialization.json"
GPU_LOCK_ROOT = PROJECT_ROOT / "runs/.gpu_locks"

AUTHORIZATION_PATH = (
    PROJECT_ROOT / "experiments/IRSTD_DCSPG_TEST_SELECTED_V1_AUTHORIZATION.json"
)
# The adapter is part of the immutable training source identity, so its
# original launch-capability literal remains stable across watcher-only
# repairs.  The watcher validates the current additive authorization below.
FORMAL_AUTHORIZATION_SHA256: str | None = "5cdea5fbba9f9da2e3533fab739f1608493bdd34de98f29c693cfdc1f94694ae"
CURRENT_FORMAL_AUTHORIZATION_SHA256: str | None = "62816a39fb7834f3e95fba0862401ce48499a87431ac81dd3b94d379c18a2a81"

# The first authorized launch committed this exact immutable ledger before the
# live output root was created.  A later watcher-only bootstrap repair must not
# rewrite that historical proof.  Resume may accept it only byte-for-byte;
# genuinely fresh launches still require the current authorization payload.
INITIAL_LAUNCH_AUTHORIZATION_SHA256 = (
    "5cdea5fbba9f9da2e3533fab739f1608493bdd34de98f29c693cfdc1f94694ae"
)
INITIAL_LAUNCH_ADAPTER_SHA256 = (
    "7292be0d82db9e15259adbad38b7b4ffaa54a89fcfb067fab356a77da13ec72b"
)
INITIAL_LAUNCH_SOURCE_ALLOWLIST_SHA256 = (
    "dbda888c0198f51cd14134fdbfe6164fedb4721709ca5a27f488f77c19316486"
)
INITIAL_LAUNCH_TASK_MANIFEST_SHA256 = (
    "6b80afb84b24b883e1bb417d43f8f6029a816422de9e015452a91a3239e510f9"
)
INITIAL_LAUNCH_LEDGER_SHA256 = (
    "5087532a04ef287d96984ced0661894a9a90900de739d6705b2a9b8bfc09aec6"
)

# Exact runner runtime closure.  The adapter entry is the normalized digest
# because its authorization literal is injected only after the final audit.
FROZEN_RUNTIME_SOURCE_SHA256: dict[str, str] = {
    "train_irstd_dcspg_ablation_test_selected_v1.py": "bc0c8830c02984b97d5fa8014811a67e228518673f0cae5f5d363e00f6a42bab",
    "experiments/IRSTD_DCSPG_ABLATION_TEST_SELECTED_V1_PROTOCOL.md": "b4922ac6ddfc05f12689a33c0f8681427c8ee3762da4f849e0891c84528c55ed",
    "experiments/irstd_dcspg_ablation_test_selected_v1_rules.json": "e92dfd8774d415b9e1aa51355a69dd9621e20d216dcf271eae81b08755666932",
    "experiments/irstd_cp_hf_s2_dcspg_v1.py": "70f9354b01bed31b6c64f7fe17d7bd0b9159ccee9887f0b9e636e508b240960d",
    "tests/test_irstd_cp_hf_s2_dcspg_v1.py": "2917f0040fc37344a1279458c3c3a96208dcd2c63193b83ecaba7394edb3f17c",
    "run_irstd_dcspg_test_selected_v1.py#normalized_authorization_literal": "3459495f77d2b8bf6662db9177832ad89df2dcc22d1ada779bf969d6c30da445",
    "tests/test_irstd_dcspg_test_selected_adapter.py": "fac4c162591d21b094d8612f297aa30afa0440808cafdb0e598704a4059f8b4e",
    "tests/test_train_irstd_dcspg_ablation_test_selected_v1.py": "5fdce67a1b761559ac28aac91a2c1ced1891e9a976f9109076c405fdf1dce7a0",
    "experiments/four_dataset_models_seed42_v1.py": "a7127fc334ea72b2021aa670f341b0d67a6070e82b9e780bd5c56ed555d0a4d3",
    "train.py": "93b2c537e127c1363429516eb616c4874613703e12fe083737191e19dbf2925e",
    "test.py": "1331f3769546e62251731986b9cddf173676be6151c38866ed63378087ea7680",
    "train_validation_selected.py": "6f870677f7724054fc8bc71e07ddde46ed3d316eb837f953be50fe8c99c6e0ed",
    "experiments/evisirst_data.py": "93f004cfbd0ed80cb04a76728d7cae4bc4317c011a1f31e2ff38ecd26589a3f1",
    "experiments/evisirst_v2_data.py": "9c6cfabf33c9c16a99574da6d87c9318800b981e585a5603ce257250fbc2b620",
    "experiments/evisirst_v2_splits.py": "e6805f9c27106982effdc36bb8965836ae14625f51fe39a2a10816a3d030b1dc",
    "experiments/three_dataset_v2_protocol.py": "c6aa18250e23f718a4284de0eff8862edff9339d092398d0a6f84a523d0495be",
    "model/EviSIRST.py": "09c3f74910eb2ee3ddd454546bf3e21e7c463a8ff2740514c8f4f3c640204b55",
    "model/__init__.py": "e9a5828ff0d7933df343c3e621e9cb46e69ddb1182afc97d389e5363cb419c2e",
    "experiments/irstd_cp_hf_s2_v1.py": "0dba0021dfe8f46d8c4e2932b478ce03609ab14be73b20c8b77817dce240436a",
    "model/_internal/Config.py": "b7e3e67c379ef4638605ebe612336b0c3cdb1a97f4d6fe731dec80b4847d5596",
    "model/_internal/SCTransNet.py": "5fb7ce711f190ead2bfcc910d2971266b2561e643c9f8a524d2032ffd48c0aeb",
    "model/_internal/tpd_clean.py": "ca61b4035bc32c99d5862e08c48096e4892e90c0de1d1f3bdf12d683c94597f2",
    "model/_internal/tpd_clean_v8_mprs_dch.py": "39e7b1618ea9f594f4abecc284afd0d845132218e2fab3657dbb957c78703c19",
    "model/_internal/tpd_forward_contract.py": "136dc2fc91276f78cfce0eb5a55e61e7bacb904ffae326bd2b61aa2fc357a94f",
    "model/_internal/tpd_frequency_gate.py": "5290bf2fd13516ea0726103dd3661fbf1ef585098b8fe933f38972887a78f4a5",
    "model/_internal/tpd_frequency_gate_v2_croa.py": "38a60bc43c75029e85ede5041420a8aae48dbba30aa4ae3a9470d31136281002",
    "model/_internal/tpd_ner_v8_mprs_dch.py": "d0d9dd226c64689f118b5d179c5415a82d26a6c803337a0f02aaaba789491acd",
    "model/_internal/tpd_ner_v8_mprs_dch_v2.py": "b6a666b00d9f53ad153089a80861907e579ec97398c66a542d20cffded03fa6b",
    "model/_internal/tpd_ner_v8_mprs_dch_v3.py": "f8fd2e307e29fd6ff26d15a7b7a1daaee22e245bc2721d68c29d34e7c653d291",
    "model/_internal/tpd_ner_v8_mprs_dch_v4_tail_aware.py": "ce64dbbdfba76cdf8f3f2f331e1ac60d4e76e23a053d73a83b785e9ca1edf37f",
    "model/_internal/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py": "c8c7db1fc8b3e83c45ee11dcb45f7c09dd2f4456554c4267aaba3b369394ff53",
    "model/_internal/tpd_ner_v8_mprs_dch_v4_tail_aware_survival.py": "3ff5c5a383384c727ea2eeb628b6121ee54f41c3cfa90362c411b7e70236703a",
    "model/_internal/tpd_query_frequency_bridge.py": "a8a6cbe18917590349e8d354d1966382414175fccabc09225c342627b975e580",
    "model/_internal/tpd_relay.py": "7b69d8e7cc62a5883355c516bdc0d30ca37036613d96d4a8ba54f51ee8744345",
    "model/_internal/tpd_sctransnet.py": "cf887b55411da8219f32b8c2a5e248b5ffec403f13146a6a1806fec4e1c014af",
    "model/_internal/tpd_survival.py": "32903ddab6da5ebc1cc941240890bbd4c4c5dcbcd34515a4b36ccd9ddbf33e40",
}
# Filled only after this test file is final.  Until then formal preflight is
# deliberately sealed off even though the 36-file runtime closure is frozen.
FROZEN_WATCHER_TEST_SHA256: str | None = (
    "08b9af79634610756a5eaa7d652ce591b9988c0880b583d1cb299f883b6a347a"
)

AUTHORIZATION_SCHEMA = "evisirst_irstd_dcspg_test_selected_authorization/v1"
MANIFEST_SCHEMA = "evisirst_irstd_dcspg_test_selected_watcher_manifest/v1"
COMPLETION_SCHEMA = "evisirst_irstd_dcspg_test_selected_watcher_completion/v1"
PREFLIGHT_SCHEMA = "evisirst_irstd_dcspg_test_selected_readonly_preflight/v1"

FINAL_EPOCH = 1000
TEST_BEGIN = 501
EXPECTED_TRAIN_RECORDS = 1000
EXPECTED_TEST_RECORDS = 500
PUBLISHED_FILENAMES = {
    "best_miou": "EviSIRST_best_mIoU.pth.tar",
    "best_pd": "EviSIRST_best_Pd.pth.tar",
}
POLL_SECONDS = 10
GPU_CONFIRM_SECONDS = 10
GPU_MEMORY_LIMIT_MIB = 1024
TERM_TIMEOUT_SECONDS = 60

ENV_PREFIX = "EVISIRST_DCSPG_TEST_SELECTED_"
METHOD_ENV = ENV_PREFIX + "METHOD_ID"
ROUTE_ENV = ENV_PREFIX + "ROUTE"
RESUME_ENV = ENV_PREFIX + "RESUME"
GPU_UUID_ENV = ENV_PREFIX + "GPU_UUID"
GPU_BUS_ENV = ENV_PREFIX + "GPU_BUS_ID"
SINGLETON_FD_ENV = ENV_PREFIX + "SINGLETON_LOCK_FD"
TASK_FD_ENV = ENV_PREFIX + "TASK_LOCK_FD"
GPU_FD_ENV = ENV_PREFIX + "GPU_LOCK_FD"

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_UUID_RE = re.compile(
    r"^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)


class DCSPGWatcherError(RuntimeError):
    """The fixed two-arm queue cannot be supervised safely."""


class _TerminationRequested(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(f"termination signal {signum}")
        self.signum = signum


class _TerminationGate:
    def __init__(self) -> None:
        self.signum: int | None = None

    def request(self, signum: int, _frame: object) -> None:
        if self.signum is None:
            self.signum = signum

    def raise_if_requested(self) -> None:
        if self.signum is not None:
            raise _TerminationRequested(self.signum)


def _no_termination_requested() -> None:
    return None


@dataclass(frozen=True)
class Task:
    method_id: str
    route: str
    gpu_index: int
    gpu_uuid: str
    gpu_bus_id: str

    @property
    def run_dir(self) -> Path:
        return (
            FORMAL_OUTPUT_ROOT
            / "formal"
            / self.method_id
            / "IRSTD-1K"
            / "run_seed_42"
        )

    @property
    def task_lock_path(self) -> Path:
        return TASK_LOCK_ROOT / f"{self.method_id}.lock"

    @property
    def gpu_lock_path(self) -> Path:
        return GPU_LOCK_ROOT / f"{self.gpu_uuid}.lock"


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    bus_id: str
    memory_used_mib: int


@dataclass
class RunningJob:
    task: Task
    resume: bool
    process: subprocess.Popen[bytes]
    gpu: GPU
    task_lock_fd: int
    gpu_lock_fd: int
    singleton_lock_fd: int
    log_handle: object
    log_path: Path


TASKS = (
    Task(
        method_id="dcspg_original",
        route="dcspg_original",
        gpu_index=0,
        gpu_uuid="GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70",
        gpu_bus_id="00000000:16:00.0",
    ),
    Task(
        method_id="dcspg_farbg",
        route="dcspg_farbg",
        gpu_index=1,
        gpu_uuid="GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640",
        gpu_bus_id="00000000:27:00.0",
    ),
)
METHOD_IDS = tuple(task.method_id for task in TASKS)


def task_for_method(method_id: str) -> Task:
    matches = [task for task in TASKS if task.method_id == method_id]
    if len(matches) != 1:
        raise DCSPGWatcherError("unsupported formal method ID")
    return matches[0]


def launch_contract(method_id: str, *, resume: bool) -> dict[str, object]:
    """Return the adapter's exact immutable capability contract."""

    if type(resume) is not bool:
        raise DCSPGWatcherError("resume must be one boolean")
    task = task_for_method(method_id)
    return {
        "method_id": task.method_id,
        "route": task.route,
        "resume": resume,
        "gpu_uuid": task.gpu_uuid,
        "gpu_bus_id": task.gpu_bus_id,
        "singleton_lock_path": WATCHER_LOCK.resolve(strict=False),
        "task_lock_path": task.task_lock_path.resolve(strict=False),
        "gpu_lock_path": task.gpu_lock_path.resolve(strict=False),
    }


def adapter_argv(method_id: str, *, resume: bool = False) -> tuple[str, ...]:
    task_for_method(method_id)
    if type(resume) is not bool:
        raise DCSPGWatcherError("resume must be one boolean")
    result = [
        os.fspath(PYTHON_BIN),
        os.fspath(ADAPTER_SOURCE),
        "--method-id",
        method_id,
    ]
    if resume:
        result.append("--resume")
    return tuple(result)


def timed_adapter_argv(method_id: str, *, resume: bool = False) -> tuple[str, ...]:
    return (os.fspath(TIME_BIN), "-v", *adapter_argv(method_id, resume=resume))


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
        raise DCSPGWatcherError("value is not strict canonical JSON") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DCSPGWatcherError(f"cannot open regular file: {path}") from exc
    digest = hashlib.sha256()
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise DCSPGWatcherError(f"not a regular file: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise DCSPGWatcherError(f"not a regular JSON file: {path}")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise DCSPGWatcherError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(token: str) -> object:
        raise DCSPGWatcherError(f"non-finite JSON constant: {token}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DCSPGWatcherError(f"malformed JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise DCSPGWatcherError("JSON root must be one object")
    _canonical_bytes(payload)
    return payload


def _normalize_authorization_literals(source: bytes) -> bytes:
    pattern = re.compile(
        rb"(?P<name>(?:CURRENT_)?FORMAL_AUTHORIZATION_SHA256)"
        rb": str \| None = (?:None|\"[0-9a-f]{64}\")"
    )

    def replace(match: re.Match[bytes]) -> bytes:
        return (
            match.group("name")
            + b': str | None = "<AUTHORIZATION_SHA256>"'
        )

    normalized, count = pattern.subn(replace, source)
    if count not in {1, 2}:
        raise DCSPGWatcherError("authorization literal normalization differs")
    return normalized


def _normalized_authorization_source_sha256(path: Path) -> str:
    return hashlib.sha256(
        _normalize_authorization_literals(_read_regular_bytes(path))
    ).hexdigest()


def frozen_source_allowlist() -> dict[str, str]:
    if len(FROZEN_RUNTIME_SOURCE_SHA256) != 36:
        raise DCSPGWatcherError("runtime source closure must contain exactly 36 files")
    result = dict(FROZEN_RUNTIME_SOURCE_SHA256)
    result[WATCHER_TEST_SOURCE.relative_to(PROJECT_ROOT).as_posix()] = (
        FROZEN_WATCHER_TEST_SHA256  # type: ignore[assignment]
    )
    for relative, digest in result.items():
        if not isinstance(digest, str) or _SHA_RE.fullmatch(digest) is None:
            raise DCSPGWatcherError(f"source allow-list is not sealed: {relative}")
    return result


def _read_regular_bytes(path: Path, *, root: Path | None = None) -> bytes:
    if root is not None:
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise DCSPGWatcherError("frozen source escapes project root") from exc
        current = root
        for component in relative.parts:
            current /= component
            if current.is_symlink():
                raise DCSPGWatcherError("frozen source path contains a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DCSPGWatcherError(f"frozen source cannot be opened: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DCSPGWatcherError(f"frozen source is not regular: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise DCSPGWatcherError("frozen source changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _source_digest(path: Path, *, normalized: bool, root: Path) -> str:
    source = _read_regular_bytes(path, root=root)
    if normalized:
        source = _normalize_authorization_literals(source)
    return hashlib.sha256(source).hexdigest()


def assert_frozen_sources(
    *,
    project_root: Path = PROJECT_ROOT,
    source_allowlist: Mapping[str, str] | None = None,
) -> None:
    allowlist = (
        frozen_source_allowlist()
        if source_allowlist is None
        else dict(source_allowlist)
    )
    runtime_keys = set(FROZEN_RUNTIME_SOURCE_SHA256)
    if source_allowlist is None and set(allowlist) != {
        *runtime_keys,
        WATCHER_TEST_SOURCE.relative_to(PROJECT_ROOT).as_posix(),
    }:
        raise DCSPGWatcherError("frozen source allow-list path set differs")
    internal_expected = {
        relative.removesuffix("#normalized_authorization_literal")
        for relative in allowlist
        if relative.removesuffix("#normalized_authorization_literal").startswith(
            "model/_internal/"
        )
    }
    internal_root = project_root / "model/_internal"
    if internal_root.is_symlink() or not internal_root.is_dir():
        raise DCSPGWatcherError("model/_internal source root differs")
    internal_observed = {
        path.relative_to(project_root).as_posix()
        for path in internal_root.glob("*.py")
    }
    if internal_observed != internal_expected:
        raise DCSPGWatcherError("model/_internal runtime path set differs")
    for relative, expected in allowlist.items():
        normalized = relative.endswith("#normalized_authorization_literal")
        source_relative = relative.removesuffix("#normalized_authorization_literal")
        pure = PurePosixPath(source_relative)
        if (
            pure.is_absolute()
            or pure.as_posix() != source_relative
            or ".." in pure.parts
            or source_relative in {"", "."}
            or not isinstance(expected, str)
            or _SHA_RE.fullmatch(expected) is None
        ):
            raise DCSPGWatcherError("frozen source entry is malformed")
        path = project_root.joinpath(*pure.parts)
        observed = _source_digest(path, normalized=normalized, root=project_root)
        if observed != expected:
            raise DCSPGWatcherError(f"frozen source differs: {relative}")


def task_manifest() -> dict[str, object]:
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "IMPLEMENTED_NOT_AUTHORIZED",
        "task_count": 2,
        "tasks": [
            {
                "method_id": task.method_id,
                "route": task.route,
                "architecture_seed": 42,
                "run_seed": 42,
                "epochs": FINAL_EPOCH,
                "train_history_count": EXPECTED_TRAIN_RECORDS,
                "test_begin_epoch": TEST_BEGIN,
                "test_end_epoch": FINAL_EPOCH,
                "test_every": 1,
                "test_history_count": EXPECTED_TEST_RECORDS,
                "run_dir": task.run_dir.relative_to(PROJECT_ROOT).as_posix(),
                "fresh_adapter_argv": list(adapter_argv(task.method_id)),
                "resume_adapter_argv": list(
                    adapter_argv(task.method_id, resume=True)
                ),
                "pinned_gpu_index": task.gpu_index,
                "pinned_gpu_uuid": task.gpu_uuid,
                "pinned_gpu_bus_id": task.gpu_bus_id,
                "published_checkpoint_roles": list(PUBLISHED_FILENAMES),
                "published_weight_filenames": dict(PUBLISHED_FILENAMES),
            }
            for task in TASKS
        ],
        "fixed_architecture_seed": 42,
        "fixed_run_seed": 42,
        "other_methods_allowed": False,
        "other_seeds_allowed": False,
        "target_output_root": FORMAL_OUTPUT_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "target_output_root_must_be_initially_absent": True,
        "quarantined_output_reused": False,
        "quarantined_output_inspected": False,
        "quarantined_output_relative_path": QUARANTINED_OUTPUT_ROOT.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "official_test_first_epoch": TEST_BEGIN,
        "official_test_selection_is_optimistic": True,
        "unbiased_test_claim_supported": False,
        "stable_over_baseline_claim_supported": False,
        "published_weight_file_count_per_arm": 2,
    }


def authorization_is_valid() -> bool:
    expected = CURRENT_FORMAL_AUTHORIZATION_SHA256
    if expected is None:
        return False
    if not isinstance(expected, str) or _SHA_RE.fullmatch(expected) is None:
        raise DCSPGWatcherError("authorization SHA-256 is malformed")
    if FORMAL_AUTHORIZATION_SHA256 != INITIAL_LAUNCH_AUTHORIZATION_SHA256:
        return False
    try:
        assert_frozen_sources()
        if _sha256(ADAPTER_SOURCE) != INITIAL_LAUNCH_ADAPTER_SHA256:
            return False
        if (
            AUTHORIZATION_PATH.is_symlink()
            or not AUTHORIZATION_PATH.is_file()
            or _sha256(AUTHORIZATION_PATH) != expected
        ):
            return False
        payload = _strict_json(AUTHORIZATION_PATH)
        allowlist_sha = _canonical_sha256(frozen_source_allowlist())
        watcher_sha = _normalized_authorization_source_sha256(WATCHER_SOURCE)
    except (OSError, DCSPGWatcherError):
        return False
    return bool(
        set(payload)
        == {
            "schema",
            "status",
            "task_manifest_sha256",
            "watcher_normalized_sha256",
            "source_allowlist_sha256",
            "target_output_root_initially_absent",
            "quarantined_output_reused",
            "adapter_normalized_sha256",
        }
        and payload.get("schema") == AUTHORIZATION_SCHEMA
        and payload.get("status") == "AUTHORIZED"
        and payload.get("task_manifest_sha256") == _canonical_sha256(task_manifest())
        and payload.get("watcher_normalized_sha256") == watcher_sha
        and payload.get("source_allowlist_sha256") == allowlist_sha
        and payload.get("adapter_normalized_sha256")
        == _normalized_authorization_source_sha256(ADAPTER_SOURCE)
        and payload.get("target_output_root_initially_absent") is True
        and payload.get("quarantined_output_reused") is False
    )


def assert_initial_target_absent(path: Path | None = None) -> None:
    """Require a genuinely new live root without touching quarantine."""

    if path is None:
        path = FORMAL_OUTPUT_ROOT
    if not path.is_absolute() or path.resolve(strict=False) != path:
        raise DCSPGWatcherError("formal output root is not canonical absolute")
    if path == QUARANTINED_OUTPUT_ROOT or path.name.startswith(
        "test_selected_v1_invalid_pre_audit_"
    ):
        raise DCSPGWatcherError("quarantined output is forbidden")
    if path.is_symlink() or path.exists():
        raise DCSPGWatcherError("new formal output root must be absent")
    current = PROJECT_ROOT
    for component in path.relative_to(PROJECT_ROOT).parts[:-1]:
        current /= component
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise DCSPGWatcherError("formal output parent chain is unsafe")


def _initialization_payload() -> dict[str, object]:
    expected = CURRENT_FORMAL_AUTHORIZATION_SHA256
    if not isinstance(expected, str) or _SHA_RE.fullmatch(expected) is None:
        raise DCSPGWatcherError("formal authorization is not frozen")
    return {
        "schema": "evisirst_irstd_dcspg_test_selected_initialization/v1",
        "authorization_sha256": expected,
        "task_manifest_sha256": _canonical_sha256(task_manifest()),
        "source_allowlist_sha256": _canonical_sha256(frozen_source_allowlist()),
        "target_output_root": os.fspath(FORMAL_OUTPUT_ROOT),
        "target_output_root_initially_absent": True,
        "quarantined_output_inspected": False,
        "quarantined_output_reused": False,
        "workers_launched_before_commit": False,
        "official_test_accessed_before_commit": False,
    }


def _initial_launch_initialization_payload() -> dict[str, object]:
    return {
        "schema": "evisirst_irstd_dcspg_test_selected_initialization/v1",
        "authorization_sha256": INITIAL_LAUNCH_AUTHORIZATION_SHA256,
        "task_manifest_sha256": INITIAL_LAUNCH_TASK_MANIFEST_SHA256,
        "source_allowlist_sha256": INITIAL_LAUNCH_SOURCE_ALLOWLIST_SHA256,
        "target_output_root": os.fspath(FORMAL_OUTPUT_ROOT),
        "target_output_root_initially_absent": True,
        "quarantined_output_inspected": False,
        "quarantined_output_reused": False,
        "workers_launched_before_commit": False,
        "official_test_accessed_before_commit": False,
    }


def _initialization_is_valid(*, allow_initial_launch: bool = False) -> bool:
    try:
        if INITIALIZATION_PATH.is_symlink() or not INITIALIZATION_PATH.is_file():
            return False
        observed = _strict_json(INITIALIZATION_PATH)
        if observed == _initialization_payload():
            return True
        return bool(
            allow_initial_launch
            and observed == _initial_launch_initialization_payload()
            and _sha256(INITIALIZATION_PATH) == INITIAL_LAUNCH_LEDGER_SHA256
        )
    except (OSError, DCSPGWatcherError):
        return False


def _commit_initialization() -> None:
    payload = _initialization_payload()
    encoded = _canonical_bytes(payload) + b"\n"
    if not INITIALIZATION_PATH.exists() and not INITIALIZATION_PATH.is_symlink():
        # Close the preflight-to-commit race while the singleton is owned.
        assert_initial_target_absent()
    _ensure_safe_directory(INITIALIZATION_PATH.parent)
    if INITIALIZATION_PATH.exists() or INITIALIZATION_PATH.is_symlink():
        if not _initialization_is_valid(
            allow_initial_launch=FORMAL_OUTPUT_ROOT.is_dir()
            and not FORMAL_OUTPUT_ROOT.is_symlink()
        ):
            raise DCSPGWatcherError("formal initialization ledger differs")
        return
    descriptor, name = tempfile.mkstemp(
        prefix=f".{INITIALIZATION_PATH.name}.",
        suffix=".tmp",
        dir=INITIALIZATION_PATH.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, INITIALIZATION_PATH, follow_symlinks=False)
        except FileExistsError as exc:
            if (
                INITIALIZATION_PATH.is_symlink()
                or not INITIALIZATION_PATH.is_file()
                or INITIALIZATION_PATH.read_bytes() != encoded
            ):
                raise DCSPGWatcherError(
                    "formal initialization no-clobber race"
                ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _runtime_dependencies_are_valid() -> None:
    try:
        target = PYTHON_BIN.resolve(strict=True)
    except OSError as exc:
        raise DCSPGWatcherError("fixed Python is unavailable") from exc
    if target != EXPECTED_PYTHON or not target.is_file() or not os.access(target, os.X_OK):
        raise DCSPGWatcherError("fixed Python target differs")
    for path in (TIME_BIN, NVIDIA_SMI, WATCHER_SOURCE):
        if path.is_symlink() or not path.is_file():
            raise DCSPGWatcherError(f"fixed dependency differs: {path}")


def _read_cmdline(path: Path) -> tuple[str, ...] | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    parts = raw.rstrip(b"\0").split(b"\0") if raw else []
    try:
        return tuple(part.decode("utf-8") for part in parts)
    except UnicodeError:
        return None


def exact_argv_processes(
    argv: Sequence[str], *, proc_root: Path = PROC_ROOT
) -> tuple[int, ...]:
    expected = tuple(argv)
    result: list[int] = []
    try:
        entries = tuple(proc_root.iterdir())
    except OSError:
        return ()
    for entry in entries:
        if entry.name.isdecimal() and _read_cmdline(entry / "cmdline") == expected:
            result.append(int(entry.name))
    return tuple(sorted(result))


def _read_ppid(pid: int, *, proc_root: Path = PROC_ROOT) -> int | None:
    try:
        lines = (proc_root / str(pid) / "status").read_text(
            encoding="utf-8"
        ).splitlines()
    except (OSError, UnicodeError):
        return None
    values = [
        line.split(":", 1)[1].strip()
        for line in lines
        if line.startswith("PPid:")
    ]
    if len(values) != 1 or not values[0].isdecimal():
        return None
    return int(values[0])


def assert_no_forbidden_processes(
    *,
    authorized_jobs: Sequence[RunningJob] = (),
    proc_root: Path = PROC_ROOT,
) -> None:
    """Reject unregistered workers while permitting exact supervised pairs."""

    allowed_pids = {os.getpid(), os.getppid()}
    allowed_parents: dict[int, tuple[str, ...]] = {}
    allowed_children: dict[int, tuple[str, ...]] = {}
    for job in authorized_jobs:
        if job.process.poll() is not None:
            raise DCSPGWatcherError("authorized worker already exited")
        allowed_parents[job.process.pid] = timed_adapter_argv(
            job.task.method_id, resume=job.resume
        )
        allowed_children[job.process.pid] = adapter_argv(
            job.task.method_id, resume=job.resume
        )
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as exc:
        raise DCSPGWatcherError("cannot inspect process table") from exc
    for entry in entries:
        if not entry.name.isdecimal() or int(entry.name) in allowed_pids:
            continue
        pid = int(entry.name)
        argv = _read_cmdline(entry / "cmdline")
        if argv is None or not any(
            argument in {os.fspath(ADAPTER_SOURCE), os.fspath(RUNNER_SOURCE)}
            or os.fspath(QUARANTINED_OUTPUT_ROOT) in argument
            for argument in argv
        ):
            continue
        expected_parent = allowed_parents.get(pid)
        if expected_parent is not None and argv == expected_parent:
            continue
        parent = _read_ppid(pid, proc_root=proc_root)
        expected_child = allowed_children.get(parent if parent is not None else -1)
        if expected_child is not None and argv == expected_child:
            continue
        raise DCSPGWatcherError(f"forbidden formal worker process: {entry.name}")


def _query_nvidia_smi(args: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            [os.fspath(NVIDIA_SMI), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DCSPGWatcherError("nvidia-smi query failed") from exc
    if completed.returncode != 0:
        raise DCSPGWatcherError("nvidia-smi query returned failure")
    return completed.stdout


def sample_idle_gpus() -> dict[str, GPU]:
    rows = _query_nvidia_smi(
        (
            "--query-gpu=index,uuid,pci.bus_id,memory.used",
            "--format=csv,noheader,nounits",
        )
    )
    busy_raw = _query_nvidia_smi(
        ("--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits")
    )
    busy: set[str] = set()
    for line in busy_raw.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and _UUID_RE.fullmatch(fields[0]):
            busy.add(fields[0])
    result: dict[str, GPU] = {}
    for line in rows.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise DCSPGWatcherError("GPU inventory is malformed")
        try:
            gpu = GPU(int(fields[0]), fields[1], fields[2], int(fields[3]))
        except ValueError as exc:
            raise DCSPGWatcherError("GPU inventory value is malformed") from exc
        if _UUID_RE.fullmatch(gpu.uuid) is None or gpu.uuid in result:
            raise DCSPGWatcherError("GPU UUID inventory differs")
        if gpu.uuid not in busy and gpu.memory_used_mib <= GPU_MEMORY_LIMIT_MIB:
            result[gpu.uuid] = gpu
    return result


def _lock_status_readonly(path: Path) -> str:
    if not path.exists() and not path.is_symlink():
        return "absent"
    if path.is_symlink() or not path.is_file():
        return "unsafe"
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "owned"
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return "available"
    finally:
        os.close(descriptor)


def readonly_preflight() -> dict[str, object]:
    """Perform only reads; never mkdir, open-create, log, or launch."""

    _runtime_dependencies_are_valid()
    assert_frozen_sources()
    if not authorization_is_valid():
        raise DCSPGWatcherError("formal authorization is unavailable")
    initially_absent, states = _queue_states()
    assert_no_forbidden_processes()
    idle = sample_idle_gpus()
    pinned: dict[str, bool] = {}
    for task in TASKS:
        observed = idle.get(task.gpu_uuid)
        pinned[task.method_id] = bool(
            observed is not None
            and observed.index == task.gpu_index
            and observed.bus_id.lower() == task.gpu_bus_id.lower()
        )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "formal_launch_authorized": True,
        "authorization_sha256": CURRENT_FORMAL_AUTHORIZATION_SHA256,
        "task_manifest_sha256": _canonical_sha256(task_manifest()),
        "source_allowlist_sha256": _canonical_sha256(frozen_source_allowlist()),
        "target_output_root_initially_absent": initially_absent,
        "task_states": states,
        "quarantined_output_inspected": False,
        "quarantined_output_reused": False,
        "pinned_gpus_idle": pinned,
        "singleton_lock_status": _lock_status_readonly(WATCHER_LOCK),
        "task_lock_status": {
            task.method_id: _lock_status_readonly(task.task_lock_path)
            for task in TASKS
        },
        "gpu_lock_status": {
            task.method_id: _lock_status_readonly(task.gpu_lock_path)
            for task in TASKS
        },
        "writes_performed": False,
        "workers_launched": False,
    }


def _ensure_safe_directory(path: Path) -> None:
    try:
        relative = path.relative_to(PROJECT_ROOT / "runs")
    except ValueError as exc:
        raise DCSPGWatcherError("watcher directory escapes runs root") from exc
    current = PROJECT_ROOT / "runs"
    if current.is_symlink() or (current.exists() and not current.is_dir()):
        raise DCSPGWatcherError("runs root is unsafe")
    for component in relative.parts:
        current /= component
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise DCSPGWatcherError(f"watcher directory is unsafe: {current}")
    path.mkdir(parents=True, exist_ok=True)


def _try_lock(path: Path) -> int | None:
    _ensure_safe_directory(path.parent)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise DCSPGWatcherError("lock is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            return None
        return descriptor
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _worker_env(
    task: Task,
    *,
    resume: bool,
    singleton_lock_fd: int,
    task_lock_fd: int,
    gpu_lock_fd: int,
) -> dict[str, str]:
    env = dict(os.environ)
    for name in tuple(env):
        if name.startswith("CUDA_") or name.startswith("NVIDIA_") or name.startswith(
            ENV_PREFIX
        ):
            env.pop(name, None)
    env.update(
        {
            METHOD_ENV: task.method_id,
            ROUTE_ENV: task.route,
            RESUME_ENV: "1" if resume else "0",
            GPU_UUID_ENV: task.gpu_uuid,
            GPU_BUS_ENV: task.gpu_bus_id,
            SINGLETON_FD_ENV: str(singleton_lock_fd),
            TASK_FD_ENV: str(task_lock_fd),
            GPU_FD_ENV: str(gpu_lock_fd),
            "CUDA_VISIBLE_DEVICES": task.gpu_uuid,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def claim_pinned_gpu(
    task: Task,
    *,
    sleep: Callable[[float], None] = time.sleep,
    check_termination: Callable[[], None] = _no_termination_requested,
) -> tuple[GPU, int] | None:
    def matching(sample: Mapping[str, GPU]) -> GPU | None:
        observed = sample.get(task.gpu_uuid)
        if (
            observed is None
            or observed.index != task.gpu_index
            or observed.bus_id.lower() != task.gpu_bus_id.lower()
        ):
            return None
        return observed

    check_termination()
    first = matching(sample_idle_gpus())
    sleep(GPU_CONFIRM_SECONDS)
    check_termination()
    second = matching(sample_idle_gpus())
    if first is None or second is None:
        return None
    descriptor = _try_lock(task.gpu_lock_path)
    if descriptor is None:
        return None
    try:
        third = matching(sample_idle_gpus())
        sleep(GPU_CONFIRM_SECONDS)
        check_termination()
        fourth = matching(sample_idle_gpus())
        if third is None or fourth is None:
            os.close(descriptor)
            return None
        return fourth, descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_log(path: Path) -> object:
    _ensure_safe_directory(path.parent)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise DCSPGWatcherError("log path is unsafe")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    return os.fdopen(descriptor, "ab", buffering=0)


def launch_task(
    task: Task,
    *,
    resume: bool,
    gpu: GPU,
    gpu_lock_fd: int,
    singleton_lock_fd: int,
    authorized_jobs: Sequence[RunningJob] = (),
) -> RunningJob | None:
    if not authorization_is_valid():
        os.close(gpu_lock_fd)
        return None
    task_lock_fd = _try_lock(task.task_lock_path)
    if task_lock_fd is None:
        os.close(gpu_lock_fd)
        return None
    log_handle: object | None = None
    try:
        if len({singleton_lock_fd, task_lock_fd, gpu_lock_fd}) != 3:
            raise DCSPGWatcherError("launch lock FDs are not distinct")
        assert_no_forbidden_processes(authorized_jobs=authorized_jobs)
        observed = sample_idle_gpus().get(task.gpu_uuid)
        if (
            observed is None
            or observed.index != task.gpu_index
            or observed.bus_id.lower() != task.gpu_bus_id.lower()
        ):
            os.close(task_lock_fd)
            os.close(gpu_lock_fd)
            return None
        log_path = LOG_ROOT / f"{task.method_id}.log"
        log_handle = _open_log(log_path)
        process = subprocess.Popen(
            list(timed_adapter_argv(task.method_id, resume=resume)),
            cwd=PROJECT_ROOT,
            env=_worker_env(
                task,
                resume=resume,
                singleton_lock_fd=singleton_lock_fd,
                task_lock_fd=task_lock_fd,
                gpu_lock_fd=gpu_lock_fd,
            ),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            pass_fds=(singleton_lock_fd, task_lock_fd, gpu_lock_fd),
            start_new_session=True,
        )
        return RunningJob(
            task=task,
            resume=resume,
            process=process,
            gpu=gpu,
            task_lock_fd=task_lock_fd,
            gpu_lock_fd=gpu_lock_fd,
            singleton_lock_fd=singleton_lock_fd,
            log_handle=log_handle,
            log_path=log_path,
        )
    except BaseException:
        if log_handle is not None:
            log_handle.close()
        os.close(task_lock_fd)
        os.close(gpu_lock_fd)
        raise


def _release_job(job: RunningJob) -> None:
    try:
        job.log_handle.close()
    finally:
        for name in ("task_lock_fd", "gpu_lock_fd"):
            descriptor = getattr(job, name)
            if descriptor >= 0:
                os.close(descriptor)
                setattr(job, name, -1)


def _abort_job(job: RunningJob) -> None:
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
    finally:
        _release_job(job)


def _validated_history(
    path: Path,
    *,
    method_id: str,
    role: str,
    key: str,
    epochs: Sequence[int],
    completed_epoch: int = FINAL_EPOCH,
) -> dict[str, object]:
    payload = _strict_json(path)
    history = payload.get(key)
    if (
        payload.get("data_role") != role
        or payload.get("method_id") != method_id
        or payload.get("completed_epoch") != completed_epoch
        or not isinstance(history, list)
        or len(history) != len(epochs)
    ):
        raise DCSPGWatcherError(f"{role} history contract differs")
    observed: list[int] = []
    for row in history:
        if not isinstance(row, dict) or type(row.get("epoch")) is not int:
            raise DCSPGWatcherError(f"{role} history row is malformed")
        observed.append(int(row["epoch"]))
        if role == "test" and row.get("data_role") != "test":
            raise DCSPGWatcherError("test history role differs")
    if observed != list(epochs):
        raise DCSPGWatcherError(f"{role} history cadence differs")
    return payload


def _role_key(row: Mapping[str, object], role: str) -> tuple[float, ...]:
    try:
        epoch = row["epoch"]
        if type(epoch) is not int:
            raise TypeError
        tiny_raw = row.get("tiny_pd")
        tiny = float("-inf") if tiny_raw is None else float(tiny_raw)
        if role == "best_miou":
            values = (
                float(row["miou"]),
                float(row["pd"]),
                -float(row["fa"]),
                float(row["niou"]),
                tiny,
                -float(row["test_loss"]),
                -float(epoch),
            )
        elif role == "best_pd":
            values = (
                float(row["pd"]),
                -float(row["fa"]),
                tiny,
                float(row["miou"]),
                float(row["niou"]),
                -float(row["test_loss"]),
                -float(epoch),
            )
        else:
            raise DCSPGWatcherError("unsupported checkpoint role")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise DCSPGWatcherError("test-selection metric row is malformed") from exc
    if any(not math.isfinite(value) for value in values if value != float("-inf")):
        raise DCSPGWatcherError("test-selection metric is non-finite")
    return values


def _role_key_record(row: Mapping[str, object], role: str) -> list[float | None]:
    return [value if math.isfinite(value) else None for value in _role_key(row, role)]


def probe_progress(task_or_method: Task | str) -> dict[str, object]:
    """Validate a committed in-progress history without opening test data."""

    task = (
        task_or_method
        if isinstance(task_or_method, Task)
        else task_for_method(task_or_method)
    )
    run_dir = task.run_dir
    if not run_dir.exists() and not run_dir.is_symlink():
        return {
            "method_id": task.method_id,
            "state": "fresh",
            "completed_epoch": 0,
            "training_history_count": 0,
            "test_history_count": 0,
            "resume": False,
        }
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise DCSPGWatcherError("formal run directory is unsafe")
    training_path = run_dir / "training_history.json"
    test_path = run_dir / "test_history.json"
    latest_path = run_dir / "last_training_state.pth.tar"
    allowed_root = {
        "last_training_state.pth.tar",
        "training_history.json",
        "test_history.json",
        "test_access_started.json",
        "test_access_verified.json",
        "selection_record.json",
        "summary.json",
        "candidates",
        "published_weights",
    }
    for item in run_dir.iterdir():
        if item.name not in allowed_root and not (
            item.name.startswith(".")
            and item.name.endswith(".tmp")
            and item.is_file()
            and not item.is_symlink()
        ):
            raise DCSPGWatcherError(f"unexpected run artifact: {item.name}")
        if item.is_symlink():
            raise DCSPGWatcherError("run artifact is a symlink")
    final_markers = tuple(
        run_dir / name
        for name in ("selection_record.json", "summary.json")
    )
    if any(path.exists() or path.is_symlink() for path in final_markers):
        try:
            completion = probe_completion(task)
        except DCSPGWatcherError:
            if latest_path.is_file() and not latest_path.is_symlink():
                return {
                    "method_id": task.method_id,
                    "state": "resume",
                    "completed_epoch": 1000,
                    "training_history_count": 1000,
                    "test_history_count": 500,
                    "resume": True,
                    "finalization_transaction_pending_strict_runner_validation": True,
                }
            raise
        return {**completion, "state": "complete", "resume": False}
    if not training_path.exists() and not test_path.exists():
        if latest_path.is_symlink() or (latest_path.exists() and not latest_path.is_file()):
            raise DCSPGWatcherError("recovery checkpoint is unsafe")
        if latest_path.is_file():
            return {
                "method_id": task.method_id,
                "state": "resume",
                "completed_epoch": None,
                "training_history_count": None,
                "test_history_count": None,
                "resume": True,
            }
        substantive = [
            item
            for item in run_dir.iterdir()
            if not (item.is_dir() and item.name in {"candidates", "published_weights"})
        ]
        if substantive:
            raise DCSPGWatcherError("fresh-created run contains partial artifacts")
        return {
            "method_id": task.method_id,
            "state": "fresh",
            "completed_epoch": 0,
            "training_history_count": 0,
            "test_history_count": 0,
            "resume": False,
        }
    if not training_path.is_file() or training_path.is_symlink():
        if latest_path.is_file() and not latest_path.is_symlink():
            return {
                "method_id": task.method_id,
                "state": "resume",
                "completed_epoch": None,
                "training_history_count": None,
                "test_history_count": None,
                "resume": True,
            }
        raise DCSPGWatcherError("training history is partially committed")
    if not test_path.is_file() or test_path.is_symlink():
        if latest_path.is_file() and not latest_path.is_symlink():
            return {
                "method_id": task.method_id,
                "state": "resume",
                "completed_epoch": None,
                "training_history_count": None,
                "test_history_count": None,
                "resume": True,
            }
        raise DCSPGWatcherError("test history is partially committed")
    try:
        training = _strict_json(training_path)
        completed = training.get("completed_epoch")
        if type(completed) is not int or not 1 <= completed <= FINAL_EPOCH:
            raise DCSPGWatcherError("progress completed epoch is invalid")
        _validated_history(
            training_path,
            method_id=task.method_id,
            role="train",
            key="training_history",
            epochs=range(1, completed + 1),
            completed_epoch=completed,
        )
        test_epochs = (
            range(TEST_BEGIN, completed + 1) if completed >= TEST_BEGIN else ()
        )
        test_payload = _validated_history(
            test_path,
            method_id=task.method_id,
            role="test",
            key="test_history",
            epochs=test_epochs,
            completed_epoch=completed,
        )
    except DCSPGWatcherError:
        # A legal epoch transaction can leave latest one commit ahead of one
        # or both sidecars.  Do not weaken the trainer's validation: merely
        # route this exact allowed layout through its strict --resume path.
        if latest_path.is_file() and not latest_path.is_symlink():
            return {
                "method_id": task.method_id,
                "state": "resume",
                "completed_epoch": None,
                "training_history_count": None,
                "test_history_count": None,
                "resume": True,
                "sidecar_transaction_pending_strict_runner_validation": True,
            }
        raise
    expected_test_count = max(0, completed - TEST_BEGIN + 1)
    started = run_dir / "test_access_started.json"
    verified = run_dir / "test_access_verified.json"
    if completed < TEST_BEGIN - 1:
        if (
            test_payload.get("test_access_started") is not False
            or test_payload.get("test_access_verified") is not False
            or started.exists()
            or started.is_symlink()
            or verified.exists()
            or verified.is_symlink()
        ):
            raise DCSPGWatcherError("official test was accessed before epoch 501")
    elif completed == TEST_BEGIN - 1:
        # A normal epoch-500 boundary has neither ledger.  A crash during the
        # lazy epoch-501 activation may have committed started (and possibly
        # verified) before the epoch-501 recovery transaction.  The runner's
        # strict --resume path authenticates their full contents.
        if verified.exists() and not started.exists():
            raise DCSPGWatcherError("verified test access lacks started ledger")
        for path in (started, verified):
            if path.exists() or path.is_symlink():
                if path.is_symlink() or not path.is_file():
                    raise DCSPGWatcherError("test-access ledger is unsafe")
    elif any(path.is_symlink() or not path.is_file() for path in (started, verified)):
        raise DCSPGWatcherError("post-501 progress lacks test-access ledgers")
    if completed == FINAL_EPOCH:
        try:
            completion = probe_completion(task)
        except DCSPGWatcherError:
            if latest_path.is_file() and not latest_path.is_symlink():
                return {
                    "method_id": task.method_id,
                    "state": "resume",
                    "completed_epoch": FINAL_EPOCH,
                    "training_history_count": FINAL_EPOCH,
                    "test_history_count": EXPECTED_TEST_RECORDS,
                    "resume": True,
                    "finalization_transaction_pending_strict_runner_validation": True,
                }
            raise
        return {**completion, "state": "complete", "resume": False}
    return {
        "method_id": task.method_id,
        "state": "resume",
        "completed_epoch": completed,
        "training_history_count": completed,
        "test_history_count": expected_test_count,
        "resume": True,
    }


def _queue_states() -> tuple[bool, dict[str, dict[str, object]]]:
    """Inspect only the live root; never enumerate or stat quarantine."""

    if not FORMAL_OUTPUT_ROOT.exists() and not FORMAL_OUTPUT_ROOT.is_symlink():
        assert_initial_target_absent()
        if INITIALIZATION_PATH.exists() or INITIALIZATION_PATH.is_symlink():
            if not _initialization_is_valid():
                raise DCSPGWatcherError("formal initialization ledger differs")
        return True, {
            task.method_id: probe_progress(task)
            for task in TASKS
        }
    if FORMAL_OUTPUT_ROOT.is_symlink() or not FORMAL_OUTPUT_ROOT.is_dir():
        raise DCSPGWatcherError("live formal output root is unsafe")
    if not _initialization_is_valid(allow_initial_launch=True):
        raise DCSPGWatcherError(
            "existing live output lacks its fresh-root initialization ledger"
        )
    allowed_top = {".locks", "formal"}
    for item in FORMAL_OUTPUT_ROOT.iterdir():
        if item.name not in allowed_top or item.is_symlink() or not item.is_dir():
            raise DCSPGWatcherError("live formal output layout differs")
    lock_root = FORMAL_OUTPUT_ROOT / ".locks"
    if lock_root.exists():
        expected_locks = {f"{method}_seed_42.lock" for method in METHOD_IDS}
        for item in lock_root.iterdir():
            if (
                item.name not in expected_locks
                or item.is_symlink()
                or not item.is_file()
            ):
                raise DCSPGWatcherError("runner lock layout differs")
    formal = FORMAL_OUTPUT_ROOT / "formal"
    if formal.exists():
        for item in formal.iterdir():
            if item.name not in METHOD_IDS or item.is_symlink() or not item.is_dir():
                raise DCSPGWatcherError("formal method layout differs")
        for task in TASKS:
            method_root = formal / task.method_id
            if not method_root.exists():
                continue
            children = tuple(method_root.iterdir())
            if any(
                item.name != "IRSTD-1K" or item.is_symlink() or not item.is_dir()
                for item in children
            ):
                raise DCSPGWatcherError("formal dataset layout differs")
            dataset_root = method_root / "IRSTD-1K"
            if dataset_root.exists():
                if any(
                    item.name != "run_seed_42"
                    or item.is_symlink()
                    or not item.is_dir()
                    for item in dataset_root.iterdir()
                ):
                    raise DCSPGWatcherError("formal seed layout differs")
    return False, {task.method_id: probe_progress(task) for task in TASKS}


def _validate_offline_data_identity(runner: object, value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DCSPGWatcherError("run data identity is malformed")
    data = dict(value)
    rules = runner._load_rules()  # type: ignore[attr-defined]
    frozen = rules["data"]
    expected = {
        "dataset_root": "/home/ly/SCTransNet_main/datasets",
        "split_root": "/home/ly/EviSIRST_main/splits/v2",
        "dataset": "IRSTD-1K",
        "training_dataset_class": "experiments.evisirst_data.EviSIRSTTrainDataset",
        "training_target_rule": "raw_mask_div_255",
        "train_count": 800,
        "full_train_count": 800,
        "v2_train_count": 640,
        "v2_val_count": 160,
        "v2_train_val_union_equals_source_train": True,
        "independent_validation_split_in_this_runner": False,
        "split_manifest_sha256": runner.EXPECTED_SPLIT_MANIFEST_SHA256,  # type: ignore[attr-defined]
        "source_train_data_tree_sha256": runner.EXPECTED_SOURCE_TRAIN_DATA_TREE_SHA256,  # type: ignore[attr-defined]
        "source_train_data_tree_verified": True,
        "train_index_relative_path": "IRSTD-1K/img_idx/train_IRSTD-1K.txt",
        "train_index_file_sha256": runner.EXPECTED_TRAIN_INDEX_FILE_SHA256,  # type: ignore[attr-defined]
        "train_ordered_ids_sha256": runner.EXPECTED_TRAIN_ORDERED_IDS_SHA256,  # type: ignore[attr-defined]
        "normalization": {"mean": 87.4661865234375, "std": 39.71953201293945},
        "expected_test_contract": {
            "test_count": frozen["expected_test_count"],
            "test_index_file_sha256": frozen["expected_test_index_file_sha256"],
            "test_ordered_ids_sha256": frozen["expected_test_ordered_ids_sha256"],
            "test_image_mask_tree_sha256": frozen["expected_test_image_mask_tree_sha256"],
            "test_image_mask_tree_hash_algorithm": "ordered_length_prefixed(role,sample_id,relative_path,file_sha256)",
        },
        "test_access_is_lazy": True,
        "test_index_opened_at_identity_construction": False,
        "startup_test_index_opened": False,
        "test_split_accessed_during_preflight": False,
    }
    if data != expected:
        raise DCSPGWatcherError("run data identity differs from frozen contract")
    return data


def _validate_offline_test_identity(
    runner: object,
    value: object,
    *,
    data_identity: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DCSPGWatcherError("observed test identity is malformed")
    expected_contract = data_identity["expected_test_contract"]
    assert isinstance(expected_contract, Mapping)
    expected = {
        "test_count": 201,
        "full_test_count": 201,
        "test_index_relative_path": "IRSTD-1K/img_idx/test_IRSTD-1K.txt",
        "test_index_file_sha256": expected_contract["test_index_file_sha256"],
        "test_ordered_ids_sha256": expected_contract["test_ordered_ids_sha256"],
        "test_image_mask_tree_sha256": expected_contract["test_image_mask_tree_sha256"],
        "normalization": dict(data_identity["normalization"]),
        "test_index_opened": True,
        "test_split_accessed": True,
        "first_access_epoch": 501,
    }
    observed = dict(value)
    if observed != expected:
        raise DCSPGWatcherError("observed test identity differs from frozen contract")
    return observed


def _offline_runner_contract(
    task: Task,
    identity: Mapping[str, object],
) -> tuple[object, object, dict[str, object], dict[str, object]]:
    try:
        runner = importlib.import_module("train_irstd_dcspg_ablation_test_selected_v1")
    except Exception as exc:
        raise DCSPGWatcherError("frozen offline runner cannot be imported") from exc
    if Path(runner.__file__).resolve(strict=True) != RUNNER_SOURCE:
        raise DCSPGWatcherError("offline runner module path differs")
    try:
        validated_identity = runner._validate_identity(identity)
        rules = runner._load_rules()
        method = runner._method_contract(
            SimpleNamespace(method_id=task.method_id, smoke=False), rules
        )
        model, _metadata, architecture_validation, architecture_identity = (
            runner._build_architecture(method)
        )
        data_identity = _validate_offline_data_identity(
            runner, validated_identity.get("data_identity")
        )
        source_manifest = runner._source_manifest(formal=False)
        raw_expected_paths = {
            relative.removesuffix("#normalized_authorization_literal")
            for relative in FROZEN_RUNTIME_SOURCE_SHA256
        }
        if (
            set(source_manifest.get("files", {})) != raw_expected_paths
            or len(source_manifest.get("files", {})) != 36
            or validated_identity.get("source_manifest") != source_manifest
        ):
            raise DCSPGWatcherError("run identity source manifest differs")
        args = SimpleNamespace(
            method_id=task.method_id,
            architecture_seed=42,
            run_seed=42,
            epochs=1000,
            batch_size=16,
            workers=0,
            base_lr=1e-3,
            min_lr=1e-5,
            warmup_epochs=10,
            test_begin=501,
            test_every=1,
            device="cuda:0",
            smoke=False,
            smoke_id="fixture",
        )
        expected_identity = runner._run_identity(
            args=args,
            method=method,
            architecture=architecture_identity,
            architecture_validation=architecture_validation,
            source_manifest=source_manifest,
            data_identity=data_identity,
            run_dir=task.run_dir,
        )
    except DCSPGWatcherError:
        raise
    except Exception as exc:
        raise DCSPGWatcherError("offline run identity validation failed") from exc
    if dict(validated_identity) != expected_identity:
        raise DCSPGWatcherError("run identity differs from reconstructed contract")
    return runner, model, method, expected_identity


def probe_completion(task_or_method: Task | str) -> dict[str, object]:
    """Strict offline completion proof; no dataset or test path is opened."""

    assert_frozen_sources()
    task = task_or_method if isinstance(task_or_method, Task) else task_for_method(task_or_method)
    run_dir = task.run_dir
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise DCSPGWatcherError("formal run directory is absent or unsafe")
    expected_run_entries = {
        "last_training_state.pth.tar",
        "training_history.json",
        "test_history.json",
        "test_access_started.json",
        "test_access_verified.json",
        "selection_record.json",
        "summary.json",
        "candidates",
        "published_weights",
    }
    run_entries = tuple(run_dir.iterdir())
    if (
        {path.name for path in run_entries} != expected_run_entries
        or any(path.is_symlink() for path in run_entries)
    ):
        raise DCSPGWatcherError("completed run root layout differs")
    paths = {
        "latest": run_dir / "last_training_state.pth.tar",
        "training_history": run_dir / "training_history.json",
        "test_history": run_dir / "test_history.json",
        "test_access_started": run_dir / "test_access_started.json",
        "test_access_verified": run_dir / "test_access_verified.json",
        "selection": run_dir / "selection_record.json",
        "summary": run_dir / "summary.json",
        "candidate_dir": run_dir / "candidates/test_selected",
        "published_dir": run_dir / "published_weights",
    }
    selection = _strict_json(paths["selection"])
    identity_raw = selection.get("run_identity")
    if not isinstance(identity_raw, Mapping):
        raise DCSPGWatcherError("selection lacks a run identity")
    runner, model, method, identity = _offline_runner_contract(
        task, dict(identity_raw)
    )
    expected_state = model.state_dict()
    expected_count = 577

    try:
        training_payload = _strict_json(paths["training_history"])
        test_payload = _strict_json(paths["test_history"])
        training_history = runner._validate_training_history(
            training_payload.get("training_history"),
            completed_epoch=1000,
            processed_samples=800,
            loss_kind=method["loss_kind"],
            farbg_lambda=1.0,
            total_epochs=1000,
            base_lr=1e-3,
            min_lr=1e-5,
            warmup_epochs=10,
        )
        test_history, best_epochs = runner._validate_metric_history(
            test_payload.get("test_history"),
            completed_epoch=1000,
            test_begin=501,
            test_every=1,
        )
        expected_training_payload = runner._history_payload(
            data_role="train",
            identity=identity,
            completed_epoch=1000,
            history=training_history,
            test_history_for_flags=test_history,
        )
        expected_test_payload = runner._history_payload(
            data_role="test",
            identity=identity,
            completed_epoch=1000,
            history=test_history,
            test_history_for_flags=test_history,
        )
        if training_payload != expected_training_payload or test_payload != expected_test_payload:
            raise DCSPGWatcherError("history sidecar exact contract differs")

        started, raw_test_identity = runner._validate_access_ledgers(paths, identity)
        if not started or raw_test_identity is None:
            raise DCSPGWatcherError("test-access ledgers are incomplete")
        test_identity = _validate_offline_test_identity(
            runner,
            raw_test_identity,
            data_identity=identity["data_identity"],
        )
        started_sha = _sha256(paths["test_access_started"])
        verified_sha = _sha256(paths["test_access_verified"])

        role_by_epoch: dict[int, list[str]] = {}
        for role, epoch_raw in best_epochs.items():
            if epoch_raw is None:
                raise DCSPGWatcherError("test selector returned no winner")
            role_by_epoch.setdefault(int(epoch_raw), []).append(role)
        candidate_dir = paths["candidate_dir"]
        candidate_root = run_dir / "candidates"
        if (
            candidate_root.is_symlink()
            or not candidate_root.is_dir()
            or {path.name for path in candidate_root.iterdir()} != {"test_selected"}
        ):
            raise DCSPGWatcherError("candidate root layout differs")
        if candidate_dir.is_symlink() or not candidate_dir.is_dir():
            raise DCSPGWatcherError("candidate frontier directory is unsafe")
        expected_candidate_names = {
            f"epoch_{epoch:04d}.pth.tar" for epoch in role_by_epoch
        }
        candidate_files = tuple(candidate_dir.iterdir())
        if (
            {path.name for path in candidate_files} != expected_candidate_names
            or any(path.is_symlink() or not path.is_file() for path in candidate_files)
        ):
            raise DCSPGWatcherError("candidate frontier file set differs")
        by_epoch = {int(row["epoch"]): row for row in test_history}
        frontier: dict[int, dict[str, object]] = {}
        candidate_payloads: dict[int, Mapping[str, object]] = {}
        for epoch, roles in sorted(role_by_epoch.items()):
            path = candidate_dir / f"epoch_{epoch:04d}.pth.tar"
            candidate_sha = _sha256(path)
            runner._validate_candidate(
                path,
                identity=identity,
                expected_state=expected_state,
                expected_count=expected_count,
                expected_record=by_epoch[epoch],
            )
            candidate = runner._safe_torch_load(path, label="selected candidate")
            if _sha256(path) != candidate_sha:
                raise DCSPGWatcherError("candidate changed during offline validation")
            candidate_payloads[epoch] = candidate
            frontier[epoch] = {
                "relative_path": f"candidates/test_selected/{path.name}",
                "sha256": candidate_sha,
                "roles": sorted(roles),
            }

        latest_sha = _sha256(paths["latest"])
        latest = runner._safe_torch_load(paths["latest"], label="latest recovery")
        if (
            set(latest) != runner._recovery_required_fields()
            or latest.get("schema") != runner.RECOVERY_SCHEMA
            or latest.get("model") != "EviSIRST"
            or latest.get("dataset") != "IRSTD-1K"
            or latest.get("method_id") != task.method_id
            or latest.get("epoch") != 1000
            or latest.get("run_identity") != identity
            or latest.get("training_history") != training_history
            or latest.get("test_history") != test_history
            or latest.get("best_epochs") != best_epochs
            or runner._normalize_candidate_artifacts(latest.get("candidate_artifacts"))
            != frontier
            or latest.get("test_identity") != test_identity
            or any(
                latest.get(key) != value
                for key, value in runner._phase_flags(
                    completed_epoch=1000, test_history=test_history
                ).items()
            )
        ):
            raise DCSPGWatcherError("latest recovery completion contract differs")
        runner._validate_state_dict(
            latest.get("state_dict"), expected_state, expected_count=expected_count
        )
        if _sha256(paths["latest"]) != latest_sha:
            raise DCSPGWatcherError(
                "latest recovery changed during offline validation"
            )
        if 1000 in candidate_payloads and not runner._state_tensors_equal(
            candidate_payloads[1000]["state_dict"], latest["state_dict"]
        ):
            raise DCSPGWatcherError(
                "epoch-1000 candidate differs from epoch-1000 recovery state"
            )

        published_dir = paths["published_dir"]
        if published_dir.is_symlink() or not published_dir.is_dir():
            raise DCSPGWatcherError("published weight directory is unsafe")
        published_files = tuple(published_dir.iterdir())
        if (
            {path.name for path in published_files} != set(PUBLISHED_FILENAMES.values())
            or len(published_files) != 2
            or any(path.is_symlink() or not path.is_file() for path in published_files)
        ):
            raise DCSPGWatcherError("published directory must contain exactly two weights")

        published_records: dict[str, dict[str, object]] = {}
        selection_roles: dict[str, dict[str, object]] = {}
        for role, filename in PUBLISHED_FILENAMES.items():
            epoch = int(best_epochs[role])
            artifact = frontier[epoch]
            candidate = candidate_payloads[epoch]
            state = candidate.get("state_dict")
            checkpoint_path = published_dir / filename
            checkpoint_sha = _sha256(checkpoint_path)
            expected_checkpoint = runner._checkpoint_payload(
                role=role,
                epoch=epoch,
                state=state,
                metrics=by_epoch[epoch],
                identity=identity,
                test_identity=test_identity,
                candidate_record=artifact,
                started_sha256=started_sha,
                verified_sha256=verified_sha,
            )
            observed_checkpoint = runner._safe_torch_load(
                checkpoint_path, label="published checkpoint"
            )
            runner._validate_checkpoint_payload(
                observed_checkpoint,
                expected_checkpoint,
                expected_count=expected_count,
            )
            if _sha256(checkpoint_path) != checkpoint_sha:
                raise DCSPGWatcherError(
                    "published checkpoint changed during offline validation"
                )
            checkpoint_record = {
                "relative_path": f"published_weights/{filename}",
                "sha256": checkpoint_sha,
                "epoch": epoch,
            }
            published_records[role] = checkpoint_record
            selection_roles[role] = {
                "epoch": epoch,
                "metrics": dict(by_epoch[epoch]),
                "role_key": runner.role_key_record(by_epoch[epoch], role),
                "candidate": dict(artifact),
                "published_checkpoint": dict(checkpoint_record),
            }

        expected_selection = {
            "schema": runner.SELECTION_SCHEMA,
            "status": "complete",
            "model": "EviSIRST",
            "dataset": "IRSTD-1K",
            "method_id": task.method_id,
            "run_identity": dict(identity),
            "selection_pool": identity["selector"]["pool"],
            "test_history_count": 500,
            "roles": selection_roles,
            "test_identity": dict(test_identity),
            "test_access_started_ledger_sha256": started_sha,
            "test_access_verified_ledger_sha256": verified_sha,
            "historical_test_selected_baseline_comparison_allowed": True,
            "test_used_for_structure_or_hyperparameter_selection": False,
            **runner._phase_flags(completed_epoch=1000, test_history=test_history),
        }
        if selection != expected_selection:
            raise DCSPGWatcherError("selection record exact contract differs")
        selection_sha = _sha256(paths["selection"])
        expected_summary = {
            "schema": runner.SUMMARY_SCHEMA,
            "status": "complete",
            "model": "EviSIRST",
            "dataset": "IRSTD-1K",
            "method_id": task.method_id,
            "run_identity": dict(identity),
            "completed_epoch": 1000,
            "train_count": 800,
            "training_history_count": 1000,
            "training_history_sha256": _sha256(paths["training_history"]),
            "test_history_count": 500,
            "test_history_sha256": _sha256(paths["test_history"]),
            "test_access_started_ledger_sha256": started_sha,
            "test_access_verified_ledger_sha256": verified_sha,
            "selection_record_sha256": selection_sha,
            "published_checkpoints": published_records,
            "published_weight_file_count": 2,
            "complete_six_arm_ablation_claim_supported": False,
            "test_used_for_structure_or_hyperparameter_selection": False,
            **runner._phase_flags(completed_epoch=1000, test_history=test_history),
        }
        summary = _strict_json(paths["summary"])
        if summary != expected_summary:
            raise DCSPGWatcherError("completion summary exact contract differs")
    except DCSPGWatcherError:
        raise
    except Exception as exc:
        raise DCSPGWatcherError("strict offline completion validation failed") from exc
    return {
        "schema": COMPLETION_SCHEMA,
        "method_id": task.method_id,
        "completed_epoch": 1000,
        "training_history_count": 1000,
        "test_history_count": 500,
        "test_epochs": {"first": 501, "last": 1000, "count": 500},
        "published_weight_file_count": 2,
        "published_checkpoints": published_records,
        "latest_recovery_sha256": latest_sha,
        "selection_record_sha256": selection_sha,
        "summary_sha256": _sha256(paths["summary"]),
        "runtime_source_file_count": 36,
        "runtime_source_aggregate_sha256": identity["source_manifest"]["aggregate_sha256"],
        "quarantined_output_inspected": False,
        "quarantined_output_reused": False,
    }


def _completion_path(task: Task) -> Path:
    return COMPLETION_ROOT / f"{task.method_id}.json"


def _write_completion(task: Task, payload: Mapping[str, object]) -> None:
    _ensure_safe_directory(COMPLETION_ROOT)
    path = _completion_path(task)
    encoded = _canonical_bytes(dict(payload)) + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise DCSPGWatcherError("completion ledger no-clobber conflict")
        return
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=COMPLETION_ROOT
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                raise DCSPGWatcherError("completion ledger race differs") from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def supervise(
    *,
    singleton_lock_fd: int,
    sleep: Callable[[float], None] = time.sleep,
    check_termination: Callable[[], None] = _no_termination_requested,
) -> int:
    running: dict[str, RunningJob] = {}
    try:
        if not authorization_is_valid():
            raise DCSPGWatcherError("formal authorization is unavailable")
        _initially_absent, states = _queue_states()
        for task in TASKS:
            state = states[task.method_id]
            if state.get("state") == "complete":
                _write_completion(task, probe_completion(task))
                continue
            if state.get("state") not in {"fresh", "resume"}:
                raise DCSPGWatcherError(
                    f"unsupported queue state: {task.method_id}"
                )
            check_termination()
            claimed = claim_pinned_gpu(
                task, sleep=sleep, check_termination=check_termination
            )
            if claimed is None:
                raise DCSPGWatcherError(f"pinned GPU is not idle: {task.method_id}")
            gpu, gpu_fd = claimed
            job = launch_task(
                task,
                resume=state.get("resume") is True,
                gpu=gpu,
                gpu_lock_fd=gpu_fd,
                singleton_lock_fd=singleton_lock_fd,
                authorized_jobs=tuple(running.values()),
            )
            if job is None:
                raise DCSPGWatcherError(f"failed to launch: {task.method_id}")
            running[task.method_id] = job

        while running:
            check_termination()
            for method_id, job in tuple(running.items()):
                returncode = job.process.poll()
                if returncode is None:
                    continue
                if returncode != 0:
                    _release_job(job)
                    del running[method_id]
                    raise DCSPGWatcherError(
                        f"formal worker failed: {method_id} rc={returncode}"
                    )
                completion = probe_completion(job.task)
                _write_completion(job.task, completion)
                _release_job(job)
                del running[method_id]
            if running:
                sleep(POLL_SECONDS)
        return 0
    finally:
        for job in tuple(running.values()):
            try:
                _abort_job(job)
            except BaseException:
                pass
        running.clear()


def _supervise_with_signals(*, singleton_lock_fd: int) -> int:
    gate = _TerminationGate()
    previous: list[tuple[int, object]] = []
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous.append((signum, signal.signal(signum, gate.request)))
        try:
            return supervise(
                singleton_lock_fd=singleton_lock_fd,
                check_termination=gate.raise_if_requested,
            )
        except _TerminationRequested as exc:
            return 128 + exc.signum
    finally:
        for signum, handler in reversed(previous):
            signal.signal(signum, handler)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--formal", action="store_true")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.manifest:
        print(json.dumps(task_manifest(), sort_keys=True, indent=2))
        return 0
    if args.preflight:
        print(json.dumps(readonly_preflight(), sort_keys=True, indent=2))
        return 0
    # All fail-closed checks occur before mkdir/open-create/Popen.
    readonly_preflight()
    singleton = _try_lock(WATCHER_LOCK)
    if singleton is None:
        return 0
    try:
        # This no-clobber ledger is committed before either adapter can create
        # the live output root.  It is the restart proof that the root was
        # absent at the first authorized launch.
        _commit_initialization()
        return _supervise_with_signals(singleton_lock_fd=singleton)
    finally:
        os.close(singleton)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DCSPGWatcherError",
    "FORMAL_AUTHORIZATION_SHA256",
    "FORMAL_OUTPUT_ROOT",
    "METHOD_IDS",
    "QUARANTINED_OUTPUT_ROOT",
    "TASKS",
    "Task",
    "adapter_argv",
    "assert_initial_target_absent",
    "authorization_is_valid",
    "frozen_source_allowlist",
    "launch_contract",
    "main",
    "probe_completion",
    "readonly_preflight",
    "task_for_method",
    "task_manifest",
    "timed_adapter_argv",
]
