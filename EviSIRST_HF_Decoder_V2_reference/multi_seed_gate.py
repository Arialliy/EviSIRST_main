#!/usr/bin/env python3
"""Hierarchical paired bootstrap gate for baseline/candidate IRSTD runs."""
from __future__ import annotations

import argparse
import json
import math
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
class Metrics:
    miou: float
    pd: float
    fa: float
    false_objects_per_image: float


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b > 0 else (1.0 if a == 0 else float("inf"))


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED_FIELDS.difference(row)
            if missing:
                raise ValueError(f"{path}:{line_no} missing {sorted(missing)}")
            sample_id = str(row["sample_id"])
            if sample_id in rows:
                raise ValueError(f"duplicate sample_id {sample_id!r} in {path}")
            rows[sample_id] = row
    if not rows:
        raise ValueError(f"no records in {path}")
    return rows


def arrays(rows: Iterable[dict[str, Any]]) -> dict[str, np.ndarray]:
    values = list(rows)
    numeric = REQUIRED_FIELDS.difference({"sample_id"})
    return {
        key: np.asarray([float(row[key]) for row in values], dtype=np.float64)
        for key in numeric
    }


def aggregate(a: dict[str, np.ndarray], index: np.ndarray | None = None) -> Metrics:
    selected = a if index is None else {key: value[index] for key, value in a.items()}
    return Metrics(
        miou=safe_ratio(float(selected["intersection"].sum()), float(selected["union"].sum())),
        pd=safe_ratio(
            float(selected["matched_target_count"].sum()),
            float(selected["target_count"].sum()),
        ),
        fa=safe_ratio(
            float(selected["unmatched_predicted_pixels"].sum()),
            float(selected["valid_pixels"].sum()),
        ),
        false_objects_per_image=float(
            selected["unmatched_predicted_object_count"].sum() / len(selected["union"])
        ),
    )


def geometric_mean(values: list[float]) -> float:
    if any(value < 0 for value in values):
        raise ValueError("geometric mean requires non-negative values")
    if any(value == 0 for value in values):
        return 0.0
    if any(not math.isfinite(value) for value in values):
        return float("inf")
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("pairs") if isinstance(payload, dict) else None
    if not isinstance(pairs, list) or len(pairs) < 3:
        raise ValueError("manifest must contain at least three paired seeds")
    seeds: set[int] = set()
    normalized: list[dict[str, Any]] = []
    for row in pairs:
        if not isinstance(row, dict):
            raise TypeError("each pair must be an object")
        seed = int(row["seed"])
        if seed in seeds:
            raise ValueError(f"duplicate seed {seed}")
        seeds.add(seed)
        normalized.append(
            {
                "seed": seed,
                "baseline": (path.parent / row["baseline"]).resolve(),
                "candidate": (path.parent / row["candidate"]).resolve(),
            }
        )
    return normalized


def load_pairs(manifest: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for spec in load_manifest(manifest):
        baseline_rows = read_jsonl(spec["baseline"])
        candidate_rows = read_jsonl(spec["candidate"])
        if set(baseline_rows) != set(candidate_rows):
            raise ValueError(f"paired sample IDs differ for seed {spec['seed']}")
        sample_ids = sorted(baseline_rows)
        baseline = arrays(baseline_rows[sample_id] for sample_id in sample_ids)
        candidate = arrays(candidate_rows[sample_id] for sample_id in sample_ids)
        result.append(
            {
                "seed": spec["seed"],
                "sample_ids": sample_ids,
                "baseline": baseline,
                "candidate": candidate,
                "baseline_metric": aggregate(baseline),
                "candidate_metric": aggregate(candidate),
            }
        )
    return result


def hierarchical_bootstrap(
    pairs: list[dict[str, Any]], *, draws: int, seed: int
) -> np.ndarray:
    if draws < 1000:
        raise ValueError("draws must be at least 1000")
    rng = np.random.default_rng(seed)
    seed_count = len(pairs)
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled_seed_indices = rng.integers(0, seed_count, size=seed_count)
        deltas: list[float] = []
        for seed_index in sampled_seed_indices:
            pair = pairs[int(seed_index)]
            n = len(pair["sample_ids"])
            image_index = rng.integers(0, n, size=n)
            base = aggregate(pair["baseline"], image_index)
            cand = aggregate(pair["candidate"], image_index)
            deltas.append(cand.miou - base.miou)
        values[draw] = float(np.mean(deltas))
    return values


def run(args: argparse.Namespace) -> dict[str, Any]:
    pairs = load_pairs(args.manifest)
    per_seed: list[dict[str, Any]] = []
    for pair in pairs:
        base: Metrics = pair["baseline_metric"]
        cand: Metrics = pair["candidate_metric"]
        per_seed.append(
            {
                "seed": pair["seed"],
                "sample_count": len(pair["sample_ids"]),
                "baseline": base.__dict__,
                "candidate": cand.__dict__,
                "delta_miou": cand.miou - base.miou,
                "delta_pd": cand.pd - base.pd,
                "fa_ratio": safe_ratio(cand.fa, base.fa),
                "false_objects_per_image_ratio": safe_ratio(
                    cand.false_objects_per_image, base.false_objects_per_image
                ),
            }
        )

    delta_values = [row["delta_miou"] for row in per_seed]
    pd_values = [row["delta_pd"] for row in per_seed]
    fa_ratios = [row["fa_ratio"] for row in per_seed]
    object_ratios = [row["false_objects_per_image_ratio"] for row in per_seed]
    bootstrap = hierarchical_bootstrap(
        pairs, draws=args.bootstrap_draws, seed=args.bootstrap_seed
    )
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    mean_delta = float(np.mean(delta_values))
    nonpositive_count = sum(value <= 0.0 for value in delta_values)
    checks = {
        "all_seeds_positive": nonpositive_count == 0,
        "mean_delta_miou": mean_delta >= args.min_mean_delta_miou,
        "hierarchical_ci_low": ci_low > 0.0,
        "mean_pd_safety": float(np.mean(pd_values)) >= args.min_mean_delta_pd,
        "fa_safety": geometric_mean(fa_ratios) <= args.max_fa_ratio,
        "false_object_safety": geometric_mean(object_ratios)
        <= args.max_false_object_ratio,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if not failures:
        decision = "GO"
    elif mean_delta <= 0.0 or nonpositive_count >= 2 or ci_high <= 0.0:
        decision = "STOP"
    else:
        decision = "HOLD_EXTEND_TO_5_SEEDS"

    return {
        "schema": "hf_decoder_v2_multi_seed_gate/v1",
        "seed_count": len(per_seed),
        "per_seed": per_seed,
        "aggregate": {
            "mean_delta_miou": mean_delta,
            "mean_delta_miou_percentage_points": 100.0 * mean_delta,
            "median_delta_miou": float(np.median(delta_values)),
            "mean_delta_pd": float(np.mean(pd_values)),
            "geometric_mean_fa_ratio": geometric_mean(fa_ratios),
            "geometric_mean_false_object_ratio": geometric_mean(object_ratios),
            "nonpositive_seed_count": nonpositive_count,
        },
        "bootstrap": {
            "type": "hierarchical_seed_then_image_paired_bootstrap",
            "draws": args.bootstrap_draws,
            "seed": args.bootstrap_seed,
            "mean_delta_miou_ci95": [ci_low, ci_high],
            "probability_mean_delta_gt_zero": float((bootstrap > 0.0).mean()),
        },
        "thresholds": {
            "min_mean_delta_miou": args.min_mean_delta_miou,
            "min_mean_delta_pd": args.min_mean_delta_pd,
            "max_fa_ratio": args.max_fa_ratio,
            "max_false_object_ratio": args.max_false_object_ratio,
        },
        "decision": decision,
        "failed_checks": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260817)
    parser.add_argument("--min-mean-delta-miou", type=float, default=0.003)
    parser.add_argument("--min-mean-delta-pd", type=float, default=-0.005)
    parser.add_argument("--max-fa-ratio", type=float, default=1.10)
    parser.add_argument("--max-false-object-ratio", type=float, default=1.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
