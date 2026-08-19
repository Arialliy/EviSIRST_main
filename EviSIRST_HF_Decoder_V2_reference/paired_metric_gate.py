#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REQUIRED_FIELDS = {
    "sample_id",
    "intersection",
    "union",
    "tp",
    "fp",
    "fn",
    "target_count",
    "matched_target_count",
    "unmatched_predicted_pixels",
    "unmatched_predicted_object_count",
    "valid_pixels",
}


@dataclass(frozen=True)
class Aggregate:
    miou: float
    niou: float
    precision: float
    recall: float
    f1: float
    pd: float
    fa: float
    false_objects_per_image: float


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED_FIELDS.difference(row)
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            sample_id = str(row["sample_id"])
            if sample_id in records:
                raise ValueError(f"duplicate sample_id in {path}: {sample_id}")
            records[sample_id] = row
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def _arrays(records: Iterable[dict[str, Any]]) -> dict[str, np.ndarray]:
    rows = list(records)
    numeric = REQUIRED_FIELDS.difference({"sample_id"})
    arrays = {
        field: np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        for field in numeric
    }
    return arrays


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _aggregate(a: dict[str, np.ndarray], index: np.ndarray | None = None) -> Aggregate:
    selected = a if index is None else {key: value[index] for key, value in a.items()}
    intersection = float(selected["intersection"].sum())
    union = float(selected["union"].sum())
    tp = float(selected["tp"].sum())
    fp = float(selected["fp"].sum())
    fn = float(selected["fn"].sum())
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    image_union = selected["union"]
    image_iou = np.where(
        image_union > 0,
        selected["intersection"] / np.maximum(image_union, 1.0),
        1.0,
    )
    return Aggregate(
        miou=_safe_ratio(intersection, union),
        niou=float(image_iou.mean()),
        precision=precision,
        recall=recall,
        f1=_safe_ratio(2.0 * precision * recall, precision + recall),
        pd=_safe_ratio(
            float(selected["matched_target_count"].sum()),
            float(selected["target_count"].sum()),
        ),
        fa=_safe_ratio(
            float(selected["unmatched_predicted_pixels"].sum()),
            float(selected["valid_pixels"].sum()),
        ),
        false_objects_per_image=float(
            selected["unmatched_predicted_object_count"].sum() / len(image_union)
        ),
    )


def _bootstrap_miou_delta(
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    draws: int,
    seed: int,
    chunk_size: int = 256,
) -> np.ndarray:
    if draws < 1000:
        raise ValueError("draws must be at least 1000")
    n = len(baseline["union"])
    rng = np.random.default_rng(seed)
    result = np.empty(draws, dtype=np.float64)
    cursor = 0
    while cursor < draws:
        batch = min(chunk_size, draws - cursor)
        index = rng.integers(0, n, size=(batch, n), endpoint=False)
        base_inter = baseline["intersection"][index].sum(axis=1)
        base_union = baseline["union"][index].sum(axis=1)
        cand_inter = candidate["intersection"][index].sum(axis=1)
        cand_union = candidate["union"][index].sum(axis=1)
        base_iou = np.divide(
            base_inter,
            base_union,
            out=np.ones_like(base_inter),
            where=base_union > 0,
        )
        cand_iou = np.divide(
            cand_inter,
            cand_union,
            out=np.ones_like(cand_inter),
            where=cand_union > 0,
        )
        result[cursor : cursor + batch] = cand_iou - base_iou
        cursor += batch
    return result


def _ratio(candidate: float, baseline: float) -> float:
    if baseline == 0.0:
        return 1.0 if candidate == 0.0 else float("inf")
    return candidate / baseline


def decide_gate(
    *,
    delta_miou: float,
    ci_low: float,
    ci_high: float,
    delta_pd: float,
    fa_ratio: float,
    false_object_ratio: float,
    min_delta_miou: float,
    min_delta_pd: float,
    max_fa_ratio: float,
    max_false_object_ratio: float,
) -> tuple[str, list[str]]:
    checks = {
        "delta_miou": delta_miou >= min_delta_miou,
        "bootstrap_ci_low": ci_low > 0.0,
        "pd_safety": delta_pd >= min_delta_pd,
        "fa_safety": fa_ratio <= max_fa_ratio,
        "false_object_safety": false_object_ratio <= max_false_object_ratio,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if not failures:
        return "GO", []
    if delta_miou <= 0.0 or ci_high <= 0.0:
        return "STOP", failures
    return "HOLD", failures


def run(args: argparse.Namespace) -> dict[str, Any]:
    baseline_rows = _read_jsonl(args.baseline)
    candidate_rows = _read_jsonl(args.candidate)
    if set(baseline_rows) != set(candidate_rows):
        only_baseline = sorted(set(baseline_rows) - set(candidate_rows))[:10]
        only_candidate = sorted(set(candidate_rows) - set(baseline_rows))[:10]
        raise ValueError(
            "paired sample sets differ: "
            f"only_baseline={only_baseline}, only_candidate={only_candidate}"
        )
    sample_ids = sorted(baseline_rows)
    baseline = _arrays(baseline_rows[sample_id] for sample_id in sample_ids)
    candidate = _arrays(candidate_rows[sample_id] for sample_id in sample_ids)
    baseline_metric = _aggregate(baseline)
    candidate_metric = _aggregate(candidate)

    bootstrap = _bootstrap_miou_delta(
        baseline,
        candidate,
        draws=args.bootstrap_draws,
        seed=args.bootstrap_seed,
    )
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    delta_miou = candidate_metric.miou - baseline_metric.miou
    delta_pd = candidate_metric.pd - baseline_metric.pd
    fa_ratio = _ratio(candidate_metric.fa, baseline_metric.fa)
    false_object_ratio = _ratio(
        candidate_metric.false_objects_per_image,
        baseline_metric.false_objects_per_image,
    )
    decision, failures = decide_gate(
        delta_miou=delta_miou,
        ci_low=ci_low,
        ci_high=ci_high,
        delta_pd=delta_pd,
        fa_ratio=fa_ratio,
        false_object_ratio=false_object_ratio,
        min_delta_miou=args.min_delta_miou,
        min_delta_pd=args.min_delta_pd,
        max_fa_ratio=args.max_fa_ratio,
        max_false_object_ratio=args.max_false_object_ratio,
    )
    return {
        "schema": "hf_decoder_v2_paired_gate/v1",
        "sample_count": len(sample_ids),
        "baseline": baseline_metric.__dict__,
        "candidate": candidate_metric.__dict__,
        "delta": {
            "miou": delta_miou,
            "miou_percentage_points": 100.0 * delta_miou,
            "niou": candidate_metric.niou - baseline_metric.niou,
            "f1": candidate_metric.f1 - baseline_metric.f1,
            "pd": delta_pd,
            "fa": candidate_metric.fa - baseline_metric.fa,
            "fa_ratio": fa_ratio,
            "false_objects_per_image_ratio": false_object_ratio,
        },
        "bootstrap": {
            "draws": args.bootstrap_draws,
            "seed": args.bootstrap_seed,
            "miou_delta_ci95": [ci_low, ci_high],
            "probability_delta_gt_zero": float((bootstrap > 0.0).mean()),
        },
        "thresholds": {
            "min_delta_miou": args.min_delta_miou,
            "min_delta_pd": args.min_delta_pd,
            "max_fa_ratio": args.max_fa_ratio,
            "max_false_object_ratio": args.max_false_object_ratio,
        },
        "decision": decision,
        "failed_checks": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260817)
    parser.add_argument("--min-delta-miou", type=float, default=0.003)
    parser.add_argument("--min-delta-pd", type=float, default=-0.005)
    parser.add_argument("--max-fa-ratio", type=float, default=1.10)
    parser.add_argument("--max-false-object-ratio", type=float, default=1.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
