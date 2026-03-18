# Predictive analyses (primarily Mode 4)

This folder contains scripts implementing the predictive analyses described in the EKEEL project deliverable **D4.2**.
Predictive modeling is implemented **primarily for Mode 4 (IKCM)** because Mode 4 provides explicit performance outcomes and a richer interaction trace.

## Prerequisites
Run the Mode 4 pipeline first. It must generate:
- `pipeline_mode4/outputs/Report/Mode4Features.csv`
- `pipeline_mode4/outputs/Report/ProcessedLogs/*.csv` (with an `Action` column)

## 1) Random Forest outcome prediction
```bash
python predictive/src/mode4_predict_rf.py \
  --features_csv pipeline_mode4/outputs/Report/Mode4Features.csv \
  --out_dir pipeline_mode4/outputs/Report/Predictive
```

## 2) HMM sequence modeling
The HMM script supports recent hmmlearn versions:
- uses `CategoricalHMM` when available; otherwise
- falls back to `MultinomialHMM` using one-hot observations.

```bash
python predictive/src/mode4_sequence_hmm.py \
  --processed_logs_dir pipeline_mode4/outputs/Report/ProcessedLogs \
  --features_csv pipeline_mode4/outputs/Report/Mode4Features.csv \
  --out_dir pipeline_mode4/outputs/Report/Predictive
```
