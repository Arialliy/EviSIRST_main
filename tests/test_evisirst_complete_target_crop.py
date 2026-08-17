from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage import measure

from experiments import evisirst_complete_target_crop as complete_crop
from experiments import evisirst_v2_data as v2_data
from experiments import evisirst_v2_splits as split_builder
from experiments import three_dataset_v2_protocol as source_protocol


class CompleteTargetCropTest(unittest.TestCase):
    @staticmethod
    def _plan(
        mask: np.ndarray, *, occurrence: int = 0
    ) -> complete_crop.CompleteTargetTransformPlan:
        return complete_crop.derive_complete_target_transform_plan(
            mask,
            run_seed=7,
            dataset="IRSTD-1K",
            epoch=11,
            sample_id="fixture_001",
            occurrence=occurrence,
        )

    @classmethod
    def _plan_for_request(
        cls, mask: np.ndarray, request: str
    ) -> complete_crop.CompleteTargetTransformPlan:
        for occurrence in range(100):
            plan = cls._plan(mask, occurrence=occurrence)
            if plan.requested_strategy == request:
                return plan
        raise AssertionError(f"no deterministic {request!r} request found")

    def test_policy_identity_is_frozen_json_and_forbids_test_access(self) -> None:
        identity = complete_crop.policy_identity()

        self.assertEqual(identity["request_probability"]["complete_target"], 0.5)
        self.assertEqual(complete_crop.CROP_SIZE, source_protocol.PATCH_SIZE)
        self.assertEqual(complete_crop.CROP_SIZE, 256)
        self.assertEqual(identity["complete_target_margin_pixels_each_side"], 8)
        self.assertEqual(identity["component_neighborhood"], "8-connected")
        self.assertEqual(identity["formal_num_workers"], 0)
        self.assertFalse(identity["test_split_accessed"])
        self.assertEqual(
            complete_crop.policy_identity_sha256(),
            "34431e3ed23472c65b13e64faae8d97b0bb5c45f4358d3afff5933d8e47b905e",
        )
        json.dumps(identity, allow_nan=False)

    def test_stateless_seed_binds_every_field_and_new_version(self) -> None:
        base = {
            "run_seed": 7,
            "dataset": "IRSTD-1K",
            "epoch": 0,
            "sample_id": "a",
            "occurrence": 0,
        }
        seed = complete_crop.stateless_augmentation_seed(**base)
        mutations = (
            {**base, "run_seed": 8},
            {**base, "dataset": "NUDT-SIRST"},
            {**base, "epoch": 1},
            {**base, "sample_id": "b"},
            {**base, "occurrence": 1},
        )

        self.assertEqual(seed, 1429035142915217255)
        self.assertEqual(
            len({seed, *(complete_crop.stateless_augmentation_seed(**x) for x in mutations)}),
            6,
        )
        self.assertNotEqual(
            seed,
            complete_crop.stateless_augmentation_seed(
                **base, augmentation_version="different-policy-version"
            ),
        )
        mask = np.zeros((300, 300), dtype=np.bool_)
        mask[100:103, 120:124] = True
        self.assertEqual(self._plan(mask), self._plan(mask))
        self.assertNotEqual(
            self._plan(mask).augmentation_seed,
            self._plan(mask, occurrence=1).augmentation_seed,
        )

    def test_request_draw_is_exact_half_probability_not_mask_dependent(self) -> None:
        mask = np.zeros((256, 256), dtype=np.bool_)
        plans = [
            complete_crop.derive_complete_target_transform_plan(
                mask,
                run_seed=42,
                dataset="IRSTD-1K",
                epoch=3,
                sample_id="ratio",
                occurrence=occurrence,
            )
            for occurrence in range(1000)
        ]

        complete_requests = sum(
            plan.requested_strategy == "complete_target" for plan in plans
        )
        self.assertEqual(complete_requests, 509)
        self.assertTrue(
            all(
                plan.fallback_reason == "no_target_components"
                for plan in plans
                if plan.requested_strategy == "complete_target"
            )
        )

    def test_complete_crop_has_margin_and_never_cuts_any_component(self) -> None:
        mask = np.zeros((400, 400), dtype=np.bool_)
        # The two diagonal pixels are one target only under 8-connectivity.
        mask[180, 180] = True
        mask[181, 181] = True
        mask[230:234, 245:249] = True
        plan = self._plan_for_request(mask, "complete_target")

        self.assertEqual(plan.realized_strategy, "complete_target")
        self.assertFalse(plan.fallback)
        self.assertEqual(plan.cut_component_labels, ())
        labels = measure.label(mask, connectivity=2)
        regions = {int(region.label): region for region in measure.regionprops(labels)}
        self.assertEqual(len(regions), 2)
        selected = regions[plan.selected_component_label]
        min_row, min_col, max_row, max_col = selected.bbox
        self.assertGreaterEqual(min_row - plan.crop_top, 8)
        self.assertGreaterEqual(min_col - plan.crop_left, 8)
        self.assertGreaterEqual(plan.crop_top + 256 - max_row, 8)
        self.assertGreaterEqual(plan.crop_left + 256 - max_col, 8)
        crop_labels = labels[
            plan.crop_top : plan.crop_top + 256,
            plan.crop_left : plan.crop_left + 256,
        ]
        for label, region in regions.items():
            inside = int(np.count_nonzero(crop_labels == label))
            self.assertIn(inside, (0, int(region.area)))

    def test_infeasible_complete_request_is_explicit_uniform_fallback(self) -> None:
        mask = np.zeros((300, 300), dtype=np.bool_)
        # Every 256-wide crop intersects and cuts this 300-wide component.
        mask[100, :] = True
        mask[150:152, 150:152] = True
        plan = self._plan_for_request(mask, "complete_target")

        self.assertEqual(plan.realized_strategy, "uniform")
        self.assertTrue(plan.fallback)
        self.assertEqual(
            plan.fallback_reason,
            "no_complete_target_crop_without_cut_components",
        )
        self.assertGreater(plan.complete_candidate_checks, 0)
        self.assertGreaterEqual(plan.cut_component_count, 1)
        audit = plan.audit_dict()
        self.assertEqual(audit["requested_strategy"], "complete_target")
        self.assertEqual(audit["realized_strategy"], "uniform")
        self.assertTrue(audit["fallback"])
        self.assertEqual(audit["cut_component_count"], plan.cut_component_count)
        self.assertFalse(audit["test_split_accessed"])

    def test_uniform_request_is_unconstrained_and_audits_cut_components(self) -> None:
        mask = np.zeros((300, 300), dtype=np.bool_)
        mask[120, :] = True
        plan = self._plan_for_request(mask, "uniform")

        self.assertEqual(plan.realized_strategy, "uniform")
        self.assertFalse(plan.fallback)
        self.assertGreaterEqual(plan.cut_component_count, 1)

    def test_original_flip_and_transpose_operations_are_preserved(self) -> None:
        mask = np.zeros((300, 300), dtype=np.bool_)
        base_plan = self._plan_for_request(mask, "uniform")
        plan = replace(
            base_plan,
            crop_top=3,
            crop_left=5,
            flip_axis0=True,
            flip_axis1=True,
            transpose=True,
        )
        image = np.arange(300 * 300, dtype=np.float32).reshape(300, 300)
        target = image + 1.0

        transformed_image, transformed_target = complete_crop.apply_crop_and_augment(
            image, target, plan
        )
        expected = image[3:259, 5:261][::-1, ::-1].transpose(1, 0)

        self.assertTrue(transformed_image.flags.c_contiguous)
        self.assertTrue(transformed_target.flags.c_contiguous)
        np.testing.assert_array_equal(transformed_image, expected)
        np.testing.assert_array_equal(transformed_target, expected + 1.0)

    def test_every_flip_tuple_exactly_matches_r1_v2_planner(self) -> None:
        mask = np.zeros((320, 301), dtype=np.bool_)
        mask[40:43, 50:54] = True
        mask[210:212, 260:263] = True
        for epoch in range(20):
            for occurrence in (0, 1, 7):
                with self.subTest(epoch=epoch, occurrence=occurrence):
                    plan = complete_crop.derive_complete_target_transform_plan(
                        mask,
                        run_seed=1446202191,
                        dataset="IRSTD-1K",
                        epoch=epoch,
                        sample_id="same_sample",
                        occurrence=occurrence,
                    )
                    legacy = v2_data._transform_plan(
                        dataset="IRSTD-1K",
                        sample_id="same_sample",
                        run_seed=1446202191,
                        epoch=epoch,
                        occurrence=occurrence,
                        height=mask.shape[0],
                        width=mask.shape[1],
                        binary_crop_mask=mask,
                    )
                    self.assertEqual(
                        (plan.flip_axis0, plan.flip_axis1, plan.transpose),
                        (legacy.flip_axis0, legacy.flip_axis1, legacy.transpose),
                    )
                    self.assertEqual(
                        plan.legacy_augmentation_seed, legacy.augmentation_seed
                    )
                    self.assertEqual(
                        plan.legacy_crop_attempts_before_flip,
                        legacy.crop_attempts,
                    )
                    self.assertEqual(
                        plan.legacy_augmentation_version,
                        v2_data.AUGMENTATION_VERSION,
                    )

    def test_audit_accumulator_requires_workers_zero_and_counts_all_outcomes(self) -> None:
        with self.assertRaisesRegex(
            complete_crop.CompleteTargetCropError, "exactly 0"
        ):
            complete_crop.CropAuditAccumulator(formal_num_workers=1)
        mask = np.zeros((300, 300), dtype=np.bool_)
        mask[120, :] = True
        complete_plan = self._plan_for_request(mask, "complete_target")
        uniform_plan = self._plan_for_request(mask, "uniform")
        accumulator = complete_crop.CropAuditAccumulator()
        accumulator.update(complete_plan)
        accumulator.update(uniform_plan)

        summary = accumulator.compute()
        self.assertEqual(summary["observation_count"], 2)
        self.assertEqual(summary["requested_counts"], {"complete_target": 1, "uniform": 1})
        self.assertEqual(summary["realized_counts"], {"complete_target": 0, "uniform": 2})
        self.assertEqual(summary["fallback_count"], 1)
        self.assertEqual(summary["cut_component_crop_count"], 2)
        self.assertFalse(summary["test_split_accessed"])

    def test_audit_state_is_raw_json_native_strict_and_resume_safe(self) -> None:
        mask = np.zeros((300, 300), dtype=np.bool_)
        mask[120, :] = True
        plans = (
            self._plan_for_request(mask, "complete_target"),
            self._plan_for_request(mask, "uniform"),
        )
        original = complete_crop.CropAuditAccumulator()
        for plan in plans:
            original.update(plan)
        state = original.state_dict()
        encoded = json.dumps(state, allow_nan=False, sort_keys=True)
        restored = complete_crop.CropAuditAccumulator()
        restored.load_state_dict(json.loads(encoded))

        self.assertNotIn("fallback_rate", state)
        self.assertNotIn("cut_component_crop_rate", state)
        self.assertEqual(restored.state_dict(), state)
        self.assertEqual(restored.compute(), original.compute())
        corruptions = []
        wrong_schema = json.loads(encoded)
        wrong_schema["schema"] = "wrong"
        corruptions.append(wrong_schema)
        wrong_total = json.loads(encoded)
        wrong_total["requested_counts"]["uniform"] += 1
        corruptions.append(wrong_total)
        wrong_workers = json.loads(encoded)
        wrong_workers["formal_num_workers"] = 0.0
        corruptions.append(wrong_workers)
        wrong_fallback = json.loads(encoded)
        wrong_fallback["fallback_count"] = 0
        corruptions.append(wrong_fallback)
        extra_rate = json.loads(encoded)
        extra_rate["fallback_rate"] = 0.5
        corruptions.append(extra_rate)
        for corruption in corruptions:
            with self.subTest(corruption=corruption), self.assertRaises(
                complete_crop.CompleteTargetCropError
            ):
                complete_crop.CropAuditAccumulator().load_state_dict(corruption)


class MatchedTargetDiagnosticsTest(unittest.TestCase):
    def test_metrics_have_exact_pixel_area_and_centroid_definitions(self) -> None:
        target = np.zeros((32, 32), dtype=np.float32)
        target[10:12, 10:12] = 1.0
        probability = np.zeros_like(target)
        probability[10, 10:12] = 0.9

        result = complete_crop.matched_target_diagnostics(probability, target)

        self.assertEqual(result["matched_component_count"], 1)
        self.assertEqual(result["matched_target_pixel_recall"], 0.5)
        self.assertEqual(result["matched_component_area_ratio"], 0.5)
        self.assertEqual(result["centroid_error"], 0.5)
        self.assertEqual(result["component_neighborhood"], "8-connected")
        self.assertEqual(result["match_rule"], "centroid_distance<3")

    def test_diagonal_pixels_are_one_eight_connected_component(self) -> None:
        target = np.zeros((32, 32), dtype=np.float32)
        probability = np.zeros_like(target)
        target[10, 10] = target[11, 11] = 1.0
        probability[10, 10] = probability[11, 11] = 1.0

        result = complete_crop.matched_target_diagnostics(probability, target)

        self.assertEqual(result["target_component_count"], 1)
        self.assertEqual(result["predicted_component_count"], 1)
        self.assertEqual(result["matched_component_count"], 1)
        self.assertEqual(result["matched_target_pixel_recall"], 1.0)

    def test_hungarian_is_one_to_one_and_radius_three_is_strict(self) -> None:
        target = np.zeros((32, 32), dtype=np.float32)
        probability = np.zeros_like(target)
        target[10, 10] = target[10, 14] = 1.0
        probability[10, 12] = 1.0
        one_to_one = complete_crop.matched_target_diagnostics(probability, target)

        exact_radius_target = np.zeros((32, 32), dtype=np.float32)
        exact_radius_prediction = np.zeros_like(exact_radius_target)
        exact_radius_target[5, 5] = 1.0
        exact_radius_prediction[5, 8] = 1.0
        exact_radius = complete_crop.matched_target_diagnostics(
            exact_radius_prediction, exact_radius_target
        )

        self.assertEqual(one_to_one["matched_component_count"], 1)
        self.assertEqual(exact_radius["matched_component_count"], 0)
        self.assertIsNone(exact_radius["matched_target_pixel_recall"])
        self.assertIsNone(exact_radius["matched_component_area_ratio"])
        self.assertIsNone(exact_radius["centroid_error"])

    def test_additive_diagnostics_use_micro_pixel_recall(self) -> None:
        first_target = np.zeros((20, 20), dtype=np.float32)
        first_prediction = np.zeros_like(first_target)
        first_target[2:4, 2:4] = 1.0
        first_prediction[2, 2:4] = 1.0
        second_target = np.zeros((20, 20), dtype=np.float32)
        second_prediction = np.zeros_like(second_target)
        second_target[10, 10] = 1.0
        second_prediction[10, 10] = 1.0
        metrics = complete_crop.MatchedTargetDiagnostics()
        metrics.update(first_prediction, first_target)
        metrics.update(second_prediction, second_target)

        result = metrics.compute()
        self.assertEqual(result["matched_component_count"], 2)
        self.assertEqual(result["matched_target_pixel_recall"], 3 / 5)
        self.assertEqual(result["matched_component_area_ratio"], 0.75)
        self.assertEqual(result["centroid_error"], 0.25)


class CompleteTargetDatasetAdapterTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        dataset = "NUAA-SIRST"
        dataset_root = root / "datasets"
        image_dir = dataset_root / dataset / "images"
        mask_dir = dataset_root / dataset / "masks"
        index_path = dataset_root / dataset / "img_idx" / f"train_{dataset}.txt"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        index_path.parent.mkdir(parents=True)
        sample_ids = [f"sample_{position:02d}" for position in range(10)]
        index_path.write_text("\n".join(sample_ids), encoding="utf-8")
        for position, sample_id in enumerate(sample_ids):
            image = np.full((300, 300), position + 1, dtype=np.uint8)
            mask = np.zeros((300, 300), dtype=np.uint8)
            mask[120:123, 130:133] = 255
            Image.fromarray(image).save(image_dir / f"{sample_id}.png")
            Image.fromarray(mask).save(mask_dir / f"{sample_id}.png")
        records, source_identity = split_builder.analyze_fixture_index(
            dataset_root=dataset_root,
            dataset=dataset,
            index_path=index_path,
            image_dir=image_dir,
            mask_dir=mask_dir,
        )
        bundle = split_builder.build_split_bundle(
            dataset=dataset,
            records=records,
            source_index=source_identity,
            split_seed=20260811,
            run_seed=42,
            val_fraction=0.2,
        )
        split_root = root / "splits" / "v2"
        split_builder.materialize_bundle(bundle, split_root)
        return dataset_root, split_root

    def test_train_adapter_preserves_tensor_contract_and_audits_crop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_root, split_root = self._fixture(Path(temporary))
            dataset = complete_crop.EviSIRSTCompleteTargetTrainDataset(
                "NUAA-SIRST",
                dataset_root=dataset_root,
                split_root=split_root,
                target_mode="binary",
                run_seed=9,
                return_metadata=True,
                formal_num_workers=0,
            )
            item = dataset[(0, 3)]
            repeated = dataset[(0, 3)]
            summary = dataset.crop_audit_summary()
            audit_state = dataset.crop_audit_state_dict()
            resumed = complete_crop.EviSIRSTCompleteTargetTrainDataset(
                "NUAA-SIRST",
                dataset_root=dataset_root,
                split_root=split_root,
                target_mode="binary",
                run_seed=9,
                formal_num_workers=0,
            )
            resumed.load_crop_audit_state_dict(audit_state)

        self.assertEqual(item["split"], "train")
        self.assertEqual(item["image"].shape, (1, 256, 256))
        self.assertEqual(item["mask"].shape, (1, 256, 256))
        self.assertEqual(item["image"].dtype, torch.float32)
        self.assertTrue(torch.equal(item["image"], repeated["image"]))
        self.assertTrue(torch.equal(item["mask"], repeated["mask"]))
        self.assertEqual(item["crop_audit"], repeated["crop_audit"])
        self.assertEqual(
            dataset.metadata["crop_policy_sha256"],
            complete_crop.policy_identity_sha256(),
        )
        self.assertFalse(dataset.metadata["test_index_opened"])
        self.assertEqual(summary["observation_count"], 2)
        self.assertEqual(resumed.crop_audit_state_dict(), audit_state)

    def test_train_adapter_rejects_nonzero_formal_workers_before_io(self) -> None:
        with self.assertRaisesRegex(
            complete_crop.CompleteTargetCropError, "exactly 0"
        ):
            complete_crop.EviSIRSTCompleteTargetTrainDataset(
                "NUAA-SIRST",
                dataset_root="does-not-exist",
                target_mode="binary",
                run_seed=1,
                formal_num_workers=1,
            )


if __name__ == "__main__":
    unittest.main()
