# Historical incomplete final-selection runs (superseded)

These runs are retained as historical evidence and must not be interpreted as
final model-selection results. The corrected full Ground Effect run is
workflow `33846305685`; its outputs are in this directory and its publication
visualization package is in `reports/paper_visuals/`.

## Full selection experiment

Command:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=.analysis-mpl-cache \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -u analysis/run_final_model_selection.py \
  --output-dir reports/final_model_selection
```

Status: interrupted after several minutes during the first chronological HGB
fit, while scikit-learn was binning/sorting the training matrix. No full-data
metrics or artifact were generated.

## Cliff comparison

Command:

```bash
MPLCONFIGDIR=.analysis-mpl-cache \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -u analysis/compare_cliff_accuracy.py \
  --output-dir /tmp/f1-cliff-final
```

Status: exited with code 134 during model fitting. No cliff metrics are
reported from this run.

## Successful infrastructure pilot

The two-fold, 100-row-per-event pilot completed in `/tmp/f1-final-pilot`.
It verified schema construction and report generation only; its metrics are
not suitable for final selection and were not copied into this package.
