from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import train_sctransnet_sbsc_v32_img_idx_test_selected as runner
from experiments import sbsc_v32_test_selection as selection


DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")


def _args(tmp_path: Path, dataset: str = "IRSTD-1K"):
    return runner.parse_args(
        [
            "--dataset",
            dataset,
            "--dataset-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "runs"),
        ]
    )


def _config(tmp_path: Path, dataset: str = "IRSTD-1K") -> dict:
    args = _args(tmp_path, dataset)
    counts = runner.EXPECTED_COUNTS[dataset]
    return runner._formal_config(
        args,
        train_count=counts["train"],
        test_count=counts["test"],
        train_index_sha256="a" * 64,
        test_index_sha256="b" * 64,
        normalization={"mean": 0.5, "std": 0.25},
        source_tree_sha256="c" * 64,
    )


def _metrics(
    *,
    miou: float = 0.70,
    niou: float = 0.71,
    pd: float = 0.90,
    fa: float = 2e-5,
    tiny_pd: float | None = 0.91,
    loss: float = 0.30,
) -> dict:
    return {
        "test_loss": loss,
        "miou": miou,
        "niou": niou,
        "pixel_precision": 0.80,
        "pixel_recall": 0.75,
        "pixel_f1": 0.774,
        "pd": pd,
        "tiny_pd": tiny_pd,
        "fa": fa,
        "false_objects_per_image": 0.1,
        "target_count": 10,
        "matched_target_count": 9,
        "tiny_target_count": 5,
        "matched_tiny_target_count": 4,
        "predicted_object_count": 11,
        "unmatched_predicted_object_count": 2,
        "valid_pixel_count": 65536,
    }


def _record(
    config: dict,
    epoch: int,
    **metric_overrides,
) -> dict:
    return runner._metric_row(
        _metrics(**metric_overrides),
        epoch,
        selection_identity=config["selection_identity"],
        model_state_sha256=f"{epoch:064x}",
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_formal_args_are_frozen_for_each_source_dataset(
    tmp_path: Path, dataset: str
) -> None:
    args = _args(tmp_path, dataset)
    assert args.dataset == dataset
    assert args.epochs == 1000
    assert args.selection_begin == 500
    assert args.selection_every == 1
    assert args.seed == 42
    assert args.architecture_seed == 42
    assert args.batch_size == 16
    assert args.patch_size == 256
    assert args.workers == 0
    assert args.smoke is False


def test_dataset_set_counts_and_501_epoch_cadence_are_exact() -> None:
    assert tuple(runner.SOURCE_DATASETS) == DATASETS
    assert runner.EXPECTED_COUNTS == {
        "NUAA-SIRST": {"train": 213, "test": 214},
        "NUDT-SIRST": {"train": 663, "test": 664},
        "IRSTD-1K": {"train": 800, "test": 201},
    }
    due = [
        epoch
        for epoch in range(1, runner.FORMAL_EPOCHS + 1)
        if runner.selection_due(
            epoch,
            runner.FORMAL_SELECTION_BEGIN,
            runner.FORMAL_SELECTION_EVERY,
        )
    ]
    assert due == list(range(500, 1001))
    assert len(due) == 501


@pytest.mark.parametrize(
    "override",
    (
        ("--epochs", "999"),
        ("--selection-begin", "499"),
        ("--selection-every", "2"),
        ("--seed", "7"),
        ("--architecture-seed", "7"),
        ("--batch-size", "8"),
        ("--workers", "1"),
        ("--warmup-epochs", "9"),
        ("--max-train-samples", "2"),
        ("--max-test-images", "2"),
    ),
)
def test_formal_cli_rejects_protocol_drift(
    tmp_path: Path, override: tuple[str, str]
) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--dataset",
                "IRSTD-1K",
                "--dataset-root",
                str(tmp_path),
                override[0],
                override[1],
            ]
        )


def test_cli_rejects_any_fourth_or_aggregate_dataset(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--dataset",
                "SIRST3",
                "--dataset-root",
                str(tmp_path),
            ]
        )


def test_smoke_cli_allows_only_explicit_reduced_fixture_knobs(
    tmp_path: Path,
) -> None:
    args = runner.parse_args(
        [
            "--dataset",
            "NUAA-SIRST",
            "--dataset-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--smoke",
            "--epochs",
            "2",
            "--selection-begin",
            "2",
            "--batch-size",
            "2",
            "--warmup-epochs",
            "1",
            "--max-train-samples",
            "2",
            "--max-test-images",
            "1",
        ]
    )
    assert args.smoke is True
    assert args.epochs == 2
    assert args.selection_begin == 2
    assert args.seed == 42
    assert args.architecture_seed == 42
    assert args.max_train_samples == 2
    assert args.max_test_images == 1


@pytest.mark.parametrize("dataset", DATASETS)
def test_formal_config_discloses_test_selection_and_historical_baseline(
    tmp_path: Path, dataset: str
) -> None:
    config = _config(tmp_path, dataset)
    assert config["protocol"] == runner.PROTOCOL_NAME
    assert config["model"] == "SCTransNet-C3-SBSC-V3.2"
    assert config["method"] == "sbsc_v32"
    assert config["initialization"] == "scratch_seed42"
    assert config["baseline_checkpoint_loaded"] is False
    assert config["baseline_trained_by_runner"] is False
    assert config["baseline_reference_kind"] == "historical_existing"
    assert config["training_target_rule"] == "raw_mask_div_255"
    assert config["split_manifest"]["training_target_rule"] == "raw_mask_div_255"
    assert config["selection_begin_epoch"] == 500
    assert config["selection_end_epoch"] == 1000
    assert config["selection_every"] == 1
    assert config["data_role"] == "test"
    assert config["test_split_accessed"] is True
    assert config["test_selected"] is True
    assert config["selection_is_optimistic"] is True
    assert config["unbiased_test_claim_supported"] is False
    assert config["selection_roles"] == {
        "best_miou": [
            "miou",
            "pd",
            "-fa",
            "niou",
            "tiny_pd",
            "-test_loss",
            "-epoch",
        ],
        "best_pd": [
            "pd",
            "-fa",
            "tiny_pd",
            "miou",
            "niou",
            "-test_loss",
            "-epoch",
        ],
    }


def test_formal_selection_identity_and_test_record_replay_selector(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    identity = config["selection_identity"]
    assert set(identity) == {
        "method",
        "dataset",
        "architecture_seed",
        "run_seed",
        "split_manifest_sha256",
        "run_identity_sha256",
    }
    assert identity["method"] == "sbsc_v32"
    assert identity["architecture_seed"] == 42
    assert identity["run_seed"] == 42
    assert all(
        isinstance(identity[field], str) and len(identity[field]) == 64
        for field in ("split_manifest_sha256", "run_identity_sha256")
    )

    record = _record(config, 500, tiny_pd=None)
    assert record["schema"] == selection.RECORD_SCHEMA
    assert record["data_role"] == "test"
    assert record["tinyPd"] is None
    assert record["tiny_pd_was_undefined"] is True
    assert record["model_state_sha256"] == f"{500:064x}"
    for field, expected in identity.items():
        assert record[field] == expected
    assert record["test_split_accessed"] is True
    assert record["test_selected"] is True
    assert record["selection_is_optimistic"] is True
    assert record["unbiased_test_claim_supported"] is False

    payload = selection.select_prefix(
        [record], completed_epoch=500, expected_identity=identity
    )
    assert set(payload["roles"]) == {"best_mIoU", "best_Pd"}
    assert payload["retention_frontier_epochs"] == [500]


def test_formal_history_maps_canonical_role_names_to_runner_role_names(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    history = [
        _record(config, 500, miou=0.70, pd=0.90, fa=2e-5),
        _record(config, 501, miou=0.69, pd=0.95, fa=1e-5),
    ]
    assert runner._validate_history(
        history, completed_epoch=501, config=config
    ) == {"best_miou": 500, "best_pd": 501}


def test_state_wrapper_is_order_shape_dtype_and_finite_strict(monkeypatch) -> None:
    monkeypatch.setattr(
        runner.core,
        "validate_sbsc_v32_state_dict",
        lambda value, method: None,
    )
    expected = {
        "weight": torch.zeros(2, dtype=torch.float32),
        "counter": torch.zeros(1, dtype=torch.int64),
    }
    valid = {
        "weight": torch.ones(2, dtype=torch.float32),
        "counter": torch.ones(1, dtype=torch.int64),
    }
    assert runner._validate_state(valid, expected) is valid

    reversed_order = {
        "counter": valid["counter"],
        "weight": valid["weight"],
    }
    with pytest.raises(ValueError, match="key/order"):
        runner._validate_state(reversed_order, expected)
    with pytest.raises(ValueError, match="tensor contract"):
        runner._validate_state(
            {**valid, "weight": torch.ones(3, dtype=torch.float32)}, expected
        )
    with pytest.raises(ValueError, match="non-finite"):
        runner._validate_state(
            {**valid, "weight": torch.tensor([float("nan"), 1.0])}, expected
        )


def test_role_checkpoints_preserve_state_and_have_distinct_role_contracts(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    metrics = _record(config, 500)
    state = {"weight": torch.tensor([1.0])}
    best_miou = runner._slim_checkpoint(
        state=state,
        dataset="IRSTD-1K",
        role="best_miou",
        epoch=500,
        metrics=metrics,
        config=config,
    )
    best_pd = runner._slim_checkpoint(
        state=state,
        dataset="IRSTD-1K",
        role="best_pd",
        epoch=500,
        metrics=metrics,
        config=config,
    )
    assert best_miou is not best_pd
    assert best_miou["state_dict"] is not best_pd["state_dict"]
    assert torch.equal(best_miou["state_dict"]["weight"], state["weight"])
    assert torch.equal(best_pd["state_dict"]["weight"], state["weight"])
    assert best_miou["checkpoint_role"] == "best_miou"
    assert best_pd["checkpoint_role"] == "best_pd"
    assert best_miou["selection_metric"] == "global_foreground_mIoU"
    assert best_pd["selection_metric"] == "Pd"
    for checkpoint in (best_miou, best_pd):
        assert checkpoint["test_selected"] is True
        assert checkpoint["selection_is_optimistic"] is True
        assert checkpoint["unbiased_test_claim_supported"] is False
        assert checkpoint["baseline_reference_kind"] == "historical_existing"
        assert checkpoint["training_target_rule"] == "raw_mask_div_255"


def test_run_source_fixes_candidate_method_and_two_physical_filenames() -> None:
    source_path = Path(runner.__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    builder_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "build_sctransnet_sbsc_v32_method"
    ]
    assert len(builder_calls) == 1
    method_keywords = [
        keyword.value
        for keyword in builder_calls[0].keywords
        if keyword.arg == "method"
    ]
    assert len(method_keywords) == 1
    assert isinstance(method_keywords[0], ast.Name)
    assert method_keywords[0].id == "METHOD_NAME"
    assert runner.METHOD_NAME == "sbsc_v32"

    method_cli = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--method"
    ]
    assert method_cli == []

    assert '"best_miou": run_dir / "best_mIoU.pth.tar"' in source
    assert '"best_pd": run_dir / "best_Pd.pth.tar"' in source
    assert source.count('run_dir / "best_mIoU.pth.tar"') == 1
    assert source.count('run_dir / "best_Pd.pth.tar"') == 1

    run_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    ]
    assert len(run_functions) == 1
    device_checks = [
        node
        for node in ast.walk(run_functions[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "require_device"
    ]
    assert len(device_checks) == 1


def test_source_hash_contract_contains_candidate_sources_not_baseline_tree() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    function = ast.parse(source)
    source_hash_functions = [
        node
        for node in function.body
        if isinstance(node, ast.FunctionDef) and node.name == "_source_tree_sha256"
    ]
    assert len(source_hash_functions) == 1
    literals = {
        node.value
        for node in ast.walk(source_hash_functions[0])
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "sbsc_v32_test_selection.py" in literals
    assert "sctransnet_sbsc_v32.py" in literals
    assert "baseline" not in literals
    assert "SCTransNet" not in literals
