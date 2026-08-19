from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import run_irstd_cp_hf_s2_mechanism_diagnostic_v1 as diagnostic


class CPHFS2MechanismDiagnosticContractTest(unittest.TestCase):
    def test_cli_freezes_registered_seed_and_cuda_device(self) -> None:
        args = diagnostic.parse_args(
            ["--dataset-root", "/tmp/data", "--run-seed", "42"]
        )
        self.assertEqual(args.device, "cuda:0")
        self.assertEqual(args.run_seed, 42)
        with self.assertRaises(SystemExit):
            diagnostic.parse_args(
                [
                    "--dataset-root",
                    "/tmp/data",
                    "--run-seed",
                    "42",
                    "--device",
                    "cpu",
                ]
            )
        with self.assertRaises(SystemExit):
            diagnostic.parse_args(
                ["--dataset-root", "/tmp/data", "--run-seed", "7"]
            )

    def test_programmatic_api_cannot_bypass_device_or_seed(self) -> None:
        for args in (
            SimpleNamespace(dataset_root=Path("/tmp/data"), run_seed=42, device="cpu"),
            SimpleNamespace(
                dataset_root=Path("/tmp/data"), run_seed=7, device="cuda:0"
            ),
        ):
            with self.assertRaisesRegex(Exception, "frozen seed/device"):
                diagnostic.evaluate_payload(args)

    def test_output_path_is_fixed_per_registered_seed(self) -> None:
        path = diagnostic.resolve_output_path(42)
        self.assertEqual(
            path.relative_to(diagnostic.OUTPUT_ROOT),
            Path("run_seed_42/result.json"),
        )
        with self.assertRaisesRegex(Exception, "outside the registry"):
            diagnostic.resolve_output_path(7)
        with self.assertRaisesRegex(Exception, "outside the registry"):
            diagnostic.resolve_output_path(42.0)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
