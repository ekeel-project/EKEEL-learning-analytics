# Mode 2 (ICM) — Learning Analytics pipeline

## What this pipeline does
- log parsing and session reconstruction
- feature extraction
- optional EDA (plots and descriptive summaries)
- feature selection (variance and correlation filters)
- clustering / profiling
- sequence characterization
- HTML reporting

## Run
```bash
python pipeline_mode2/src/mode2_pipeline.py \
  --log_dir pipeline_mode2/data/logs \
  --out_root pipeline_mode2/outputs \
  --skip_eda
```

Outputs are written under `pipeline_mode2/outputs/Report/`.

The folder reference_outputs/ contains pre-generated artifacts produced by running the delivered pipelines on the included log dataset. 