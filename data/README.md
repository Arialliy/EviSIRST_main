# Data files

- `evisirst_results.json`: three frozen EviSIRST final checkpoints and exact metrics.
- `baseline_results.json`: each dataset's single designated SCTransNet Baseline checkpoint.
- `model_vs_baseline.csv`: compact paper-table source; every row is one real checkpoint.
- `evisirst_evaluation/*.json`: original final-model evaluation records copied byte-for-byte.

Raw image datasets are not included. Baseline raw logs are in `../baseline/logs/`.
The three Baseline weights are in `../baseline/checkpoints/<dataset>/SCTransNet.pth.tar`;
the three EviSIRST weights are in `../results/<dataset>/EviSIRST.pth.tar`.

The JSON/CSV tables preserve the historical evaluator. Root `test.py` instead applies one
corrected common evaluator to both models, so newly rerun values—especially NUAA and legacy
Baseline Pd/Fa—need not be bitwise identical to these historical records.

Files under `evisirst_evaluation/` are byte-for-byte provenance records and may contain absolute
paths from the original experiment workspace. Those paths are historical evidence only; current
weights are loaded exclusively from `../results/<dataset>/EviSIRST.pth.tar`.
