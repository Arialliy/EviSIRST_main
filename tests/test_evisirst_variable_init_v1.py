from __future__ import annotations

import gc
import json
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments import evisirst_variable_init_v1 as variable_init
from experiments import four_dataset_models_seed42_v1 as frozen_builder
from experiments.evisirst_variable_init_v1 import (
    BASE_PARAMETER_COUNT,
    BASE_STATE_KEY_COUNT,
    CONFIRMATION_INIT_SEEDS,
    EviSIRSTVariableInitError,
    SCHEMA,
    STATUS,
    UINT32_MAX,
    initialize_variable_evisirst,
    require_uint32_seed,
    source_identity,
)
from model.EviSIRST import EviSIRST, initialize_evisirst


class EviSIRSTVariableInitializationTest(unittest.TestCase):
    def test_confirmation_seed_registry_matches_frozen_rules(self) -> None:
        rules_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "irstd_psbfr_v1_screen_rules.json"
        )
        rules = json.loads(rules_path.read_text(encoding="utf-8"))
        pairs = rules["formal_confirmation"]["init_run_seed_pairs"]
        self.assertEqual(
            CONFIRMATION_INIT_SEEDS,
            tuple(pair[0] for pair in pairs),
        )

    def test_seed_domain_is_exact_uint32_nonbool(self) -> None:
        self.assertEqual(require_uint32_seed(0), 0)
        self.assertEqual(require_uint32_seed(UINT32_MAX), UINT32_MAX)
        for invalid in (True, False, -1, UINT32_MAX + 1, 1.0, "42", None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "uint32"
            ):
                require_uint32_seed(invalid)

    def test_seed42_is_bitwise_identical_to_public_initializer(self) -> None:
        torch.manual_seed(71_009)
        caller_rng = torch.get_rng_state().clone()
        oracle, _ = initialize_evisirst(
            "IRSTD-1K", seed=42, training=True
        )
        self.assertTrue(torch.equal(torch.get_rng_state(), caller_rng))
        candidate, metadata = initialize_variable_evisirst(
            "IRSTD-1K", seed=42, training=True
        )
        self.assertTrue(torch.equal(torch.get_rng_state(), caller_rng))
        self.assertIs(type(candidate), EviSIRST)
        oracle_state = oracle.state_dict()
        candidate_state = candidate.state_dict()
        self.assertEqual(tuple(candidate_state), tuple(oracle_state))
        for key in oracle_state:
            self.assertTrue(
                torch.equal(candidate_state[key], oracle_state[key]), key
            )
        self.assertEqual(metadata["schema"], SCHEMA)
        self.assertEqual(metadata["status"], STATUS)
        self.assertEqual(metadata["architecture_seed"], 42)
        self.assertEqual(metadata["state_key_count"], BASE_STATE_KEY_COUNT)
        self.assertEqual(metadata["parameter_count"], BASE_PARAMETER_COUNT)
        self.assertFalse(hasattr(candidate, "target_survival"))
        self.assertTrue(metadata["manifest"]["temporary_patch_restored"])
        self.assertFalse(metadata["manifest"]["public_api_replacement"])
        self.assertTrue(metadata["manifest"]["confirmation_only"])
        self.assertIs(frozen_builder._require_seed, variable_init._ORIGINAL_REQUIRE_SEED)

    def test_five_confirmation_seeds_are_reproducible_and_distinct(self) -> None:
        self.assertEqual(len(CONFIRMATION_INIT_SEEDS), 5)
        self.assertEqual(
            len(CONFIRMATION_INIT_SEEDS), len(set(CONFIRMATION_INIT_SEEDS))
        )
        selected_tensors: list[torch.Tensor] = []
        state_hashes: list[str] = []
        selected_key: str | None = None
        for seed in CONFIRMATION_INIT_SEEDS:
            torch.manual_seed(2_003 + seed)
            caller_rng = torch.get_rng_state().clone()
            first, first_metadata = initialize_variable_evisirst(
                "IRSTD-1K", seed=seed, training=True
            )
            self.assertTrue(torch.equal(torch.get_rng_state(), caller_rng))
            second, second_metadata = initialize_variable_evisirst(
                "IRSTD-1K", seed=seed, training=True
            )
            self.assertTrue(torch.equal(torch.get_rng_state(), caller_rng))
            first_state = first.state_dict()
            second_state = second.state_dict()
            self.assertEqual(tuple(first_state), tuple(second_state))
            self.assertEqual(len(first_state), BASE_STATE_KEY_COUNT)
            for key in first_state:
                self.assertTrue(torch.equal(first_state[key], second_state[key]), key)
            self.assertEqual(first_metadata["architecture_seed"], seed)
            self.assertEqual(second_metadata["architecture_seed"], seed)
            self.assertEqual(
                first_metadata["selected_model_state_sha256"],
                second_metadata["selected_model_state_sha256"],
            )
            state_hashes.append(first_metadata["selected_model_state_sha256"])
            if selected_key is None:
                selected_key = next(
                    key
                    for key, value in first_state.items()
                    if value.is_floating_point() and value.numel() > 1
                )
            selected_tensors.append(first_state[selected_key].detach().clone())
            del first, second, first_state, second_state
            gc.collect()

        self.assertEqual(len(state_hashes), len(set(state_hashes)))
        self.assertIsNotNone(selected_key)
        for left in range(len(selected_tensors)):
            for right in range(left + 1, len(selected_tensors)):
                self.assertFalse(
                    torch.equal(selected_tensors[left], selected_tensors[right]),
                    f"seeds {CONFIRMATION_INIT_SEEDS[left]} and "
                    f"{CONFIRMATION_INIT_SEEDS[right]} share {selected_key}",
                )

    def test_patch_restores_exact_guard_after_builder_exception(self) -> None:
        original = frozen_builder._require_seed
        module_keys = set(vars(frozen_builder))
        torch.manual_seed(9_119)
        caller_rng = torch.get_rng_state().clone()
        with mock.patch.object(
            frozen_builder,
            "build_paper_model",
            side_effect=RuntimeError("fixture failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                initialize_variable_evisirst(
                    "IRSTD-1K", seed=CONFIRMATION_INIT_SEEDS[0]
                )
        self.assertIs(frozen_builder._require_seed, original)
        self.assertEqual(set(vars(frozen_builder)), module_keys)
        self.assertTrue(torch.equal(torch.get_rng_state(), caller_rng))

    def test_patch_context_is_nonreentrant_and_restores_outer_guard(self) -> None:
        original = frozen_builder._require_seed
        with variable_init._temporary_uint32_seed_guard():
            self.assertIs(frozen_builder._require_seed, require_uint32_seed)
            with self.assertRaisesRegex(
                EviSIRSTVariableInitError, "already active"
            ):
                initialize_variable_evisirst(
                    "IRSTD-1K", seed=CONFIRMATION_INIT_SEEDS[0]
                )
            self.assertIs(frozen_builder._require_seed, require_uint32_seed)
        self.assertIs(frozen_builder._require_seed, original)

    def test_patch_refuses_foreign_guard_and_releases_lock(self) -> None:
        original = frozen_builder._require_seed

        def foreign_guard(seed: int) -> int:
            return seed

        with mock.patch.object(frozen_builder, "_require_seed", foreign_guard):
            with self.assertRaisesRegex(
                EviSIRSTVariableInitError, "modified before entry"
            ):
                with variable_init._temporary_uint32_seed_guard():
                    self.fail("foreign guard must not be patched")
            self.assertIs(frozen_builder._require_seed, foreign_guard)
        self.assertIs(frozen_builder._require_seed, original)
        with variable_init._temporary_uint32_seed_guard():
            self.assertIs(frozen_builder._require_seed, require_uint32_seed)
        self.assertIs(frozen_builder._require_seed, original)

    def test_no_state_or_module_namespace_leak_and_eval_mode_is_explicit(self) -> None:
        original = frozen_builder._require_seed
        module_keys = set(vars(frozen_builder))
        model, metadata = initialize_variable_evisirst(
            "IRSTD-1K", seed=CONFIRMATION_INIT_SEEDS[0], training=False
        )
        self.assertIs(frozen_builder._require_seed, original)
        self.assertEqual(set(vars(frozen_builder)), module_keys)
        self.assertEqual(len(model.state_dict()), BASE_STATE_KEY_COUNT)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            BASE_PARAMETER_COUNT,
        )
        self.assertFalse(hasattr(model, "target_survival"))
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertFalse(model.training)
        self.assertEqual(model.mode, "test")
        self.assertFalse(metadata["training_mode"])
        self.assertFalse(metadata["manifest"]["training_mode"])

    def test_source_identity_binds_wrapper_and_original_builder(self) -> None:
        identity = source_identity()
        self.assertEqual(identity["schema"], variable_init.SOURCE_IDENTITY_SCHEMA)
        self.assertEqual(
            set(identity["files"]),
            {"variable_init_wrapper", "frozen_complete_builder"},
        )
        for entry in identity["files"].values():
            path = variable_init.PROJECT_ROOT / entry["relative_path"]
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())
            self.assertEqual(len(entry["sha256"]), 64)
        self.assertEqual(len(identity["source_tree_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
