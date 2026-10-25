# Streaming-classification scripts

This directory contains the project's historical synthetic streaming-classification experiment. It is intentionally separate from the 2-D quadratic lab in the repository root.

## Data

- `synthetic_covariate_shift.csv` is a 10,000-row synthetic binary stream with gradual affine feature drift.
- `extreme_shift.csv` is a 10,000-row synthetic binary stream with stronger variance multipliers.
- `generate_data.py` and `new_data.py` regenerate the two streams.

The committed CSV files are synthetic inputs, not patient records.

## Scripts

```bash
python model/generate_data.py
python model/new_data.py
python model/main_optimizer.py
python model/live_extreme.py
python model/tuning_sweep.py
```

The scripts preserve the historical HVP-norm eigen rule used for the classification prototype. That rule is **not** the same as the power-iteration plus EMA tracker used in the quadratic experiment, so classification metrics must not be presented as validation of the quadratic method.

The historical scripts also include interactive Matplotlib dashboards and were written for exploratory use. For a fully seeded repeated classification study, use the separate harness described by the paper rather than treating a single exploratory run as an uncertainty-bearing result.
