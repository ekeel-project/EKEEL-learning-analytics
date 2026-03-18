# Mode 4 (IKCM) — Learning Analytics pipeline

## What this pipeline does
- log parsing and session reconstruction
- feature extraction (including performance indicators)
- optional EDA (plots and descriptive summaries)
- feature selection (variance and correlation filters)
- clustering / profiling
- sequence characterization
- scoring and archetype indicators
- HTML reporting
- export of cleaned per-session CSV logs for sequence modeling:
  `pipeline_mode4/outputs/Report/ProcessedLogs/`

## Run
```bash
python pipeline_mode4/src/mode4_pipeline.py \
  --log_dir pipeline_mode4/data/logs \
  --out_root pipeline_mode4/outputs \
  --skip_eda
```

Outputs are written under `pipeline_mode4/outputs/Report/`.

The folder reference_outputs/ contains pre-generated artifacts produced by running the delivered pipelines on the included log dataset. 
