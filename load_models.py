"""Small public loading API for the frozen EviSIRST and SCTransNet baselines."""

from __future__ import annotations

import gc
import hashlib
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import torch

from experiments.export_three_dataset_current_bundle_v3 import (
    load_exported_current_model_v3,
)
from model.Config import get_SCTrans_config
from model.SCTransNet import SCTransNet


ROOT = Path(__file__).resolve().parent
DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")

EVISIRST_PACKAGES = {
    "NUAA-SIRST": {
        "relative_path": "results/NUAA-SIRST/EviSIRST.pth.tar",
        "epoch": 850,
        "sha256": "01031f35811ec6ca23643da32d8acdf89ddd7d5cda01f1b525a37f3c8299bf14",
    },
    "NUDT-SIRST": {
        "relative_path": "results/NUDT-SIRST/EviSIRST.pth.tar",
        "epoch": 420,
        "sha256": "497b079c89c23b26c18c8812672bf77b4a99aeabd246c77a05cea9f67fe31610",
    },
    "IRSTD-1K": {
        "relative_path": "results/IRSTD-1K/EviSIRST.pth.tar",
        "epoch": 830,
        "sha256": "87bb8bb333e28cbb053448b90d50b64eeb3bb905667451d46151079159d24bed",
    },
}

BASELINE_CHECKPOINTS = {
    "NUAA-SIRST": {
        "file": "SCTransNet.pth.tar",
        "epoch": 740,
        "sha256": "fe73b2c6ab523adbd880795d64b76f735d92f9600661edca24a3b777ce123556",
        "source_selection": "historical_best_miou",
    },
    "NUDT-SIRST": {
        "file": "SCTransNet.pth.tar",
        "epoch": 1000,
        "sha256": "5baa4e0859060228f079e97f8ce4309a71c30f829c701f3c0b63ba0f67862172",
        "source_selection": "epoch1000_user_designated_baseline",
    },
    "IRSTD-1K": {
        "file": "SCTransNet.pth.tar",
        "epoch": 713,
        "sha256": "5f702bba036f43b62fc82d349b75344f9f6c04b2b68a143311a0b48050b3371b",
        "source_selection": "historical_best_miou",
    },
}


def _require_dataset(dataset: str) -> str:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset {dataset!r}; expected one of {DATASETS}")
    return dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_evisirst(
    dataset: str,
    *,
    root: Path | str = ROOT,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Strictly validate and load one dataset-specific EviSIRST V3 package."""

    dataset = _require_dataset(dataset)
    binding = dict(EVISIRST_PACKAGES[dataset])
    package = Path(root) / binding["relative_path"]
    if not package.is_file() or package.is_symlink():
        raise FileNotFoundError(package)
    observed_sha = _sha256(package)
    if observed_sha != binding["sha256"]:
        raise ValueError(f"EviSIRST checkpoint SHA-256 differs: {package}")
    model, metadata = load_exported_current_model_v3(
        package,
        expected_dataset=dataset,
    )
    deployment = metadata.get("deployment_package")
    if not isinstance(deployment, Mapping):
        raise ValueError("EviSIRST deployment-package metadata is absent")
    if int(deployment.get("epoch", -1)) != int(binding["epoch"]):
        raise ValueError("EviSIRST checkpoint epoch differs")
    ready = dict(metadata)
    ready.update(
        {
            "dataset": dataset,
            "training_dataset": dataset,
            "evaluation_dataset": dataset,
            "checkpoint_role": "final",
            "source_selection": "historical_best_miou",
            "epoch": binding["epoch"],
            "checkpoint_path": str(package),
            "checkpoint_sha256": observed_sha,
        }
    )
    return model, ready


def load_baseline(
    dataset: str,
    *,
    root: Path | str = ROOT,
) -> tuple[SCTransNet, dict[str, Any]]:
    """Hash-check and strict-load the designated SCTransNet baseline."""

    dataset = _require_dataset(dataset)
    binding = dict(BASELINE_CHECKPOINTS[dataset])
    path = Path(root) / "baseline/checkpoints" / dataset / binding["file"]
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    observed_sha = _sha256(path)
    if observed_sha != binding["sha256"]:
        raise ValueError(f"baseline checkpoint SHA-256 differs: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or set(payload) != {
        "epoch",
        "state_dict",
        "total_loss",
    }:
        raise ValueError("baseline checkpoint payload schema differs")
    if int(payload["epoch"]) != int(binding["epoch"]):
        raise ValueError("baseline checkpoint epoch differs")
    state = payload["state_dict"]
    if not isinstance(state, Mapping) or len(state) != 510:
        raise ValueError("baseline checkpoint state must contain 510 tensors")
    if any(not isinstance(key, str) or not key.startswith("model.") for key in state):
        raise ValueError("baseline checkpoint state prefix differs")
    projected = OrderedDict((key[6:], value) for key, value in state.items())
    if any(not isinstance(value, torch.Tensor) for value in projected.values()):
        raise TypeError("baseline state values must be tensors")
    if any(
        value.is_floating_point() and not bool(torch.isfinite(value).all())
        for value in projected.values()
    ):
        raise ValueError("baseline state contains non-finite tensors")

    with redirect_stdout(StringIO()):
        model = SCTransNet(get_SCTrans_config(), mode="test", deepsuper=True)
    expected = model.state_dict()
    if set(projected) != set(expected):
        raise ValueError("baseline state key set differs from SCTransNet")
    for key, value in projected.items():
        if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError(f"baseline tensor contract differs for {key!r}")
    incompatible = model.load_state_dict(projected, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("baseline strict load returned incompatible keys")
    model.eval()
    model.mode = "test"
    metadata = {
        "model": "SCTransNet baseline",
        "dataset": dataset,
        "checkpoint_role": "baseline",
        "source_selection": binding["source_selection"],
        "epoch": binding["epoch"],
        "checkpoint_path": str(path),
        "checkpoint_sha256": observed_sha,
        "state_key_count": len(projected),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "strict_load": True,
        "mode": "test",
        "output": "sigmoid(out)",
    }
    del payload, state, expected
    gc.collect()
    return model, metadata


__all__ = [
    "BASELINE_CHECKPOINTS",
    "DATASETS",
    "EVISIRST_PACKAGES",
    "load_baseline",
    "load_evisirst",
]
