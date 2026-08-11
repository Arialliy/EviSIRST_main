"""Public EviSIRST model API.

Use :func:`load_pretrained` for normal inference.  ``EviSIRST`` is an alias of
the exact frozen 564-key network class for code that needs the architecture
type directly; its low-level constructor requires a prepared SCTransNet
parent, so most callers should use the loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
)


EviSIRST = TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet
PRETRAINED_DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
TRAINING_DATASETS = ("SIRST3", *PRETRAINED_DATASETS)
DATASETS = TRAINING_DATASETS
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def initialize_evisirst(
    dataset: str,
    *,
    seed: int = 42,
    training: bool = True,
) -> tuple[EviSIRST, dict[str, Any]]:
    """Build a clean 564-key EviSIRST graph without loading pretrained weights.

    This is the public constructor used by ``train.py``.  The historical
    training runner registered four zero-weight TSS tensors; the public graph
    deliberately has no TSS module or TSS state.
    """

    if dataset not in TRAINING_DATASETS:
        raise ValueError(
            f"unsupported dataset {dataset!r}; expected one of {TRAINING_DATASETS}"
        )
    from experiments.four_dataset_models_seed42_v1 import build_paper_model

    model, metadata = build_paper_model(
        "final",
        dataset,
        seed=seed,
        training=False,
    )
    if not isinstance(model, EviSIRST):
        raise TypeError("builder did not return the frozen EviSIRST architecture")
    if hasattr(model, "target_survival"):
        raise RuntimeError("clean EviSIRST graph unexpectedly registers TSS")
    if len(model.state_dict()) != 564:
        raise RuntimeError("clean EviSIRST graph must contain 564 state keys")
    if training:
        model.mode = "train"
        model.train()
    else:
        model.mode = "test"
        model.eval()
    ready = dict(metadata)
    ready.update(
        {
            "public_model": "EviSIRST",
            "target_survival_registered": False,
            "state_key_count": 564,
            "training_mode": bool(training),
        }
    )
    return model, ready


def load_pretrained(
    dataset: str,
    *,
    root: Path | str = PROJECT_ROOT,
) -> tuple[EviSIRST, dict[str, Any]]:
    """Validate and load the frozen dataset-specific EviSIRST V3 weight."""

    if dataset not in PRETRAINED_DATASETS:
        raise ValueError(
            "published pretrained weights exist only for "
            f"{PRETRAINED_DATASETS}; use train.py for {dataset!r}"
        )

    # Lazy import avoids a cycle because the shared loader also imports the
    # frozen SCTransNet implementation from this package.
    from load_models import load_evisirst

    model, metadata = load_evisirst(dataset, root=root)
    if not isinstance(model, EviSIRST):
        raise TypeError("loaded model is not the frozen EviSIRST architecture")
    return model, metadata


def build_evisirst(
    dataset: str,
    *,
    root: Path | str = PROJECT_ROOT,
) -> EviSIRST:
    """Return only the ready-to-run pretrained model."""

    model, _ = load_pretrained(dataset, root=root)
    return model


__all__ = [
    "DATASETS",
    "EviSIRST",
    "PRETRAINED_DATASETS",
    "TRAINING_DATASETS",
    "build_evisirst",
    "initialize_evisirst",
    "load_pretrained",
]
