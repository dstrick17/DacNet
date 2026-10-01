# DACNet reproducibility record

This directory records the final DACNet run on the shared patient split in
`../replicate_chexnet/patient_splits.csv`.

- Training code commit: `286dc7ba4d72628eb02d0c30c5dfd477b5ac8962`
- Run ID: `l7x0gzg5`
- Mean test AUROC: `0.8474449764607265`
- Mean test F1: `0.34135997328453144`
- Checkpoint SHA-256: `f965b07ef3dd2e664cb492dd71d8ef604dfe09895915320935c239df9a480529`

`test_results.json` contains the per-pathology metrics, validation-selected
thresholds, split counts, and training configuration. `run_environment.json`
records the software, hardware, command, and exact Git commit. The checkpoint
is distributed as a release asset rather than tracked in Git history.
