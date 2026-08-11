from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import torch

from experiments import export_three_dataset_current_bundle_v3 as exporter


class ThreeDatasetCurrentBundleV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_threads = int(torch.get_num_threads())
        self._original_dtype = torch.get_default_dtype()
        self._original_mkldnn = bool(torch.backends.mkldnn.enabled)

    def tearDown(self) -> None:
        torch.backends.mkldnn.enabled = self._original_mkldnn
        torch.set_default_dtype(self._original_dtype)
        if int(torch.get_num_threads()) != self._original_threads:
            torch.set_num_threads(self._original_threads)

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _token(label: str) -> str:
        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    def _assert_canonical_runtime(self) -> None:
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertEqual(torch.get_default_dtype(), torch.float32)
        self.assertTrue(torch.backends.mkldnn.enabled)
        self.assertFalse(torch.is_autocast_enabled("cpu"))

    @contextmanager
    def _external_runtime(
        self,
        *,
        threads: int,
        dtype: torch.dtype = torch.float64,
        mkldnn: bool = False,
    ):
        before_threads = int(torch.get_num_threads())
        before_dtype = torch.get_default_dtype()
        before_mkldnn = bool(torch.backends.mkldnn.enabled)
        torch.set_num_threads(threads)
        torch.set_default_dtype(dtype)
        torch.backends.mkldnn.enabled = mkldnn
        try:
            yield
            self.assertEqual(torch.get_num_threads(), threads)
            self.assertEqual(torch.get_default_dtype(), dtype)
            self.assertEqual(torch.backends.mkldnn.enabled, mkldnn)
        finally:
            torch.backends.mkldnn.enabled = before_mkldnn
            torch.set_default_dtype(before_dtype)
            if int(torch.get_num_threads()) != before_threads:
                torch.set_num_threads(before_threads)

    def _architecture(self) -> dict[str, object]:
        return {
            "public_name": exporter.MODEL_NAME,
            "public_id": exporter.MODEL_PUBLIC_ID,
            "implementation_class": (
                "TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet"
            ),
            "architecture_manifest": {"fixture": "v3"},
            "architecture_manifest_canonical_json_sha256": (
                exporter.ARCHITECTURE_MANIFEST_CANONICAL_JSON_SHA256
            ),
            "sorted_state_key_set_canonical_json_sha256": (
                exporter.SORTED_STATE_KEY_SET_CANONICAL_JSON_SHA256
            ),
            "state_key_count": exporter.INFERENCE_STATE_KEY_COUNT,
            "parameter_count": exporter.INFERENCE_PARAMETER_COUNT,
            "target_survival_registered": False,
            "public_output": "sigmoid(out)",
        }

    def _identity(self, dataset: str) -> dict[str, object]:
        return {
            "probe": exporter.CANONICAL_PROBE,
            "hash_algorithm": exporter.STATE_HASH_ALGORITHM,
            "probe_state_dict_sha256": self._token("probe"),
            "six_outputs_state_dict_sha256": self._token(
                f"six-outputs:{dataset}"
            ),
            "public_output_state_dict_sha256": self._token(
                f"public-output:{dataset}"
            ),
            "six_output_count": 6,
            "all_six_bitwise_equal": True,
            "maximum_absolute_difference": 0.0,
            "deployed_single_output_equals_sixth": True,
            "deployed_output": "sigmoid(out)",
        }

    def _contexts(self) -> dict[str, dict[str, object]]:
        architecture = self._architecture()
        contexts: dict[str, dict[str, object]] = {}
        for dataset in exporter.DATASETS:
            contexts[dataset] = {
                "dataset": dataset,
                "inference_state_dict": {
                    "fixture": torch.tensor([len(dataset)], dtype=torch.float32)
                },
                "binding": {
                    "architecture": copy.deepcopy(architecture),
                    "inference_state_sha256": self._token(f"state:{dataset}"),
                    "synthetic_identity": self._identity(dataset),
                },
            }
        return contexts

    @contextmanager
    def _mocked_v1(self):
        contexts = self._contexts()
        architecture = self._architecture()
        call_threads: dict[str, list[int]] = {
            "source": [],
            "package": [],
            "validator": [],
        }

        def record(label: str) -> None:
            call_threads[label].append(int(torch.get_num_threads()))
            self._assert_canonical_runtime()

        def validate_source(_source_root: Path, dataset: str):
            record("source")
            return copy.deepcopy(contexts[dataset])

        def make_package(context: dict[str, object]):
            record("package")
            dataset = str(context["dataset"])
            binding = context["binding"]
            assert isinstance(binding, dict)
            return {
                "schema": exporter.PACKAGE_SCHEMA,
                "dataset": dataset,
                "deployment_implementation": (
                    exporter._package_producer_implementation_binding()
                ),
                "source": {
                    "synthetic_identity": copy.deepcopy(
                        binding["synthetic_identity"]
                    )
                },
                "state_dict": copy.deepcopy(context["inference_state_dict"]),
                "synthetic_identity": copy.deepcopy(
                    binding["synthetic_identity"]
                ),
            }

        def validate_package(package_path: Path, *, expected_dataset=None):
            record("validator")
            content = Path(package_path).read_bytes()
            payload = torch.load(
                package_path,
                map_location="cpu",
                weights_only=True,
            )
            dataset = str(payload["dataset"])
            if expected_dataset is not None:
                self.assertEqual(dataset, expected_dataset)
            return {
                "schema": exporter.PACKAGE_SCHEMA,
                "dataset": dataset,
                "epoch": exporter.SOURCE_LOCKS[dataset]["epoch"],
                "path": str(Path(package_path).resolve()),
                "bytes": len(content),
                "file_sha256": hashlib.sha256(content).hexdigest(),
                "state_key_count": exporter.INFERENCE_STATE_KEY_COUNT,
                "parameter_count": exporter.INFERENCE_PARAMETER_COUNT,
                "state_sha256": contexts[dataset]["binding"][
                    "inference_state_sha256"
                ],
                "state_hash_algorithm": exporter.STATE_HASH_ALGORITHM,
                "architecture": copy.deepcopy(architecture),
                "strict_load": True,
                "tss_absent": True,
                "qfg_preserved": True,
            }

        with mock.patch.object(
            exporter.v1,
            "validate_current_source",
            side_effect=validate_source,
        ), mock.patch.object(
            exporter.v1,
            "_package_from_source",
            side_effect=make_package,
        ), mock.patch.object(
            exporter.v1,
            "validate_exported_current_package",
            side_effect=validate_package,
        ):
            yield contexts, call_threads

    def test_context_canonicalizes_and_restores_on_success_and_error(self) -> None:
        with self._external_runtime(threads=2, dtype=torch.float64, mkldnn=False):
            with exporter.canonical_synthetic_replay_context():
                self._assert_canonical_runtime()
                self.assertEqual(torch.empty(1).device.type, "cpu")
                self.assertEqual(torch.empty(1).dtype, torch.float32)
            self.assertEqual(torch.get_num_threads(), 2)
            self.assertEqual(torch.get_default_dtype(), torch.float64)
            self.assertFalse(torch.backends.mkldnn.enabled)

            with torch.autocast(
                device_type="cpu",
                enabled=True,
                dtype=torch.bfloat16,
            ):
                self.assertTrue(torch.is_autocast_enabled("cpu"))
                with exporter.canonical_synthetic_replay_context():
                    self._assert_canonical_runtime()
                self.assertTrue(torch.is_autocast_enabled("cpu"))

            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                with exporter.canonical_synthetic_replay_context():
                    self._assert_canonical_runtime()
                    raise RuntimeError("fixture failure")
            self.assertEqual(torch.get_num_threads(), 2)
            self.assertEqual(torch.get_default_dtype(), torch.float64)
            self.assertFalse(torch.backends.mkldnn.enabled)

    def test_v3_bundle_canonical_threads_hash_binding_and_idempotence(self) -> None:
        real_torch_load = torch.load
        load_kwargs: list[dict[str, object]] = []
        write_order: list[str] = []
        real_write_once = exporter.v1._write_bytes_once

        def guarded_load(*args, **kwargs):
            load_kwargs.append(dict(kwargs))
            self.assertIs(kwargs.get("weights_only"), True)
            return real_torch_load(*args, **kwargs)

        def recording_write(path: Path, content: bytes):
            write_order.append(Path(path).name)
            return real_write_once(path, content)

        with tempfile.TemporaryDirectory() as directory_text, self._mocked_v1() as mocked, mock.patch.object(
            exporter.torch,
            "load",
            side_effect=guarded_load,
        ), mock.patch.object(
            exporter.v1,
            "_write_bytes_once",
            side_effect=recording_write,
        ), mock.patch.object(
            exporter.torch,
            "set_num_interop_threads",
            side_effect=AssertionError("V3 must not set inter-op threads"),
        ):
            contexts, call_threads = mocked
            directory = Path(directory_text)
            source_root = directory / "source"
            output_root = directory / "bundle-v3"
            with self._external_runtime(
                threads=4,
                dtype=torch.float64,
                mkldnn=False,
            ):
                first = exporter.export_current_bundle(source_root, output_root)
                self.assertEqual(torch.get_num_threads(), 4)
                self.assertEqual(torch.get_default_dtype(), torch.float64)
                self.assertFalse(torch.backends.mkldnn.enabled)

            self.assertFalse(first["idempotent_existing_bundle"])
            self.assertEqual(first["schema"], exporter.BUNDLE_SCHEMA)
            self.assertEqual(write_order[-1], "COMMITTED")
            self.assertEqual(
                {path.name for path in output_root.iterdir()},
                {"packages", "manifest.json", "COMMITTED"},
            )
            self.assertTrue(all(call_threads.values()))
            self.assertTrue(
                all(
                    threads == 1
                    for calls in call_threads.values()
                    for threads in calls
                )
            )

            manifest = json.loads(
                (output_root / "manifest.json").read_text(encoding="utf-8")
            )
            committed = json.loads(
                (output_root / "COMMITTED").read_text(encoding="utf-8")
            )
            contract = exporter.synthetic_replay_execution_contract()
            self.assertEqual(contract["torch_version"], str(torch.__version__))
            self.assertTrue(contract["torch_version_bound"])
            self.assertEqual(contract["intraop_threads"], 1)
            self.assertEqual(contract["default_dtype"], "float32")
            self.assertFalse(contract["cpu_autocast_enabled"])
            self.assertTrue(contract["mkldnn_enabled"])
            self.assertFalse(contract["interop_threads_bound"])
            self.assertFalse(
                contract[
                    "arbitrary_intraop_thread_count_bitwise_equivalence_claimed"
                ]
            )
            self.assertEqual(
                manifest["synthetic_replay_execution_contract"], contract
            )
            self.assertEqual(
                committed["synthetic_replay_execution_contract"], contract
            )
            self.assertEqual(
                manifest["bundle_implementation"],
                exporter._bundle_implementation_binding(),
            )
            self.assertEqual(
                manifest["package_producer_implementation"],
                exporter._package_producer_implementation_binding(),
            )

            for entry in manifest["packages"]:
                dataset = entry["dataset"]
                package_path = output_root / entry["relative_path"]
                package = torch.load(
                    package_path,
                    map_location="cpu",
                    weights_only=True,
                )
                # The complete root identity, including all three tensor hashes,
                # is byte-for-value identical to the V1 package identity.
                self.assertEqual(
                    entry["synthetic_identity"],
                    package["synthetic_identity"],
                )
                self.assertEqual(
                    entry["synthetic_identity"],
                    contexts[dataset]["binding"]["synthetic_identity"],
                )
                self.assertEqual(
                    entry["synthetic_replay_execution_contract"], contract
                )

            before = {
                path.relative_to(output_root).as_posix(): self._sha(path)
                for path in output_root.rglob("*")
                if path.is_file()
            }
            write_count = len(write_order)
            with self._external_runtime(
                threads=2,
                dtype=torch.float64,
                mkldnn=False,
            ):
                second = exporter.export_current_bundle(source_root, output_root)
                self.assertEqual(torch.get_num_threads(), 2)
            after = {
                path.relative_to(output_root).as_posix(): self._sha(path)
                for path in output_root.rglob("*")
                if path.is_file()
            }
            self.assertTrue(second["idempotent_existing_bundle"])
            self.assertEqual(before, after)
            self.assertEqual(len(write_order), write_count)
            self.assertTrue(load_kwargs)
            self.assertTrue(
                all(call.get("weights_only") is True for call in load_kwargs)
            )

    def test_v3_loader_validates_inside_context_and_restores(self) -> None:
        calls: list[tuple[int, torch.dtype, bool]] = []
        model = torch.nn.Identity()

        def load_package(_path: Path, *, expected_dataset=None):
            calls.append(
                (
                    int(torch.get_num_threads()),
                    torch.get_default_dtype(),
                    bool(torch.backends.mkldnn.enabled),
                )
            )
            self._assert_canonical_runtime()
            self.assertEqual(expected_dataset, exporter.DATASETS[0])
            return model, {"strict_load": True}

        with mock.patch.object(
            exporter.v1,
            "load_exported_current_model",
            side_effect=load_package,
        ), mock.patch.object(
            exporter.torch,
            "set_num_interop_threads",
            side_effect=AssertionError("V3 must not set inter-op threads"),
        ), self._external_runtime(
            threads=4,
            dtype=torch.float64,
            mkldnn=False,
        ):
            loaded, metadata = exporter.load_exported_current_model_v3(
                Path("fixture.pth.tar"),
                expected_dataset=exporter.DATASETS[0],
            )
            self.assertIs(loaded, model)
            self.assertTrue(metadata["v3_canonical_package_validation"])
            self.assertEqual(torch.get_num_threads(), 4)
            self.assertEqual(torch.get_default_dtype(), torch.float64)
            self.assertFalse(torch.backends.mkldnn.enabled)
        self.assertEqual(calls, [(1, torch.float32, True)])

    def test_manifest_synthetic_identity_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text, self._mocked_v1():
            directory = Path(directory_text)
            output_root = directory / "bundle-v3"
            exporter.export_current_bundle(directory / "source", output_root)
            manifest_path = output_root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["packages"][0]["synthetic_identity"][
                "six_outputs_state_dict_sha256"
            ] = "0" * 64
            # Re-sign the root JSON: package-to-root replay binding must still
            # reject the modified synthetic hash.
            manifest["manifest_semantic_sha256"] = (
                exporter._manifest_semantic_sha256(manifest)
            )
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "synthetic identity"):
                exporter.validate_committed_bundle(output_root)

    def test_execution_contract_and_committed_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text, self._mocked_v1():
            directory = Path(directory_text)
            output_root = directory / "bundle-v3"
            exporter.export_current_bundle(directory / "source", output_root)

            committed_path = output_root / "COMMITTED"
            committed = json.loads(committed_path.read_text(encoding="utf-8"))
            committed["synthetic_replay_execution_contract"][
                "intraop_threads"
            ] = 4
            committed_path.write_text(
                json.dumps(committed, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "execution contract"):
                exporter.validate_committed_bundle(output_root)

    def test_partial_bundle_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            output_root = directory / "partial-v3"
            output_root.mkdir()
            (output_root / "partial.txt").write_text(
                "partial",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileExistsError, "incomplete or foreign"):
                exporter.export_current_bundle(directory / "source", output_root)


if __name__ == "__main__":
    unittest.main()
