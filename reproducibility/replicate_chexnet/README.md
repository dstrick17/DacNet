# CheXNet reproduction record

This directory records the final 14-pathology CheXNet reproduction and its
secondary thresholded-F1 evaluation.

- Training code commit: `745c13c`
- Threshold-evaluation code commit: `cae6fe14c2fa06236df8c8b19ff117f95c4babd4`
- Mean test AUROC: `0.8124544738644026`
- Fixed-0.5 mean test F1: `0.12709002926245724`
- Validation-thresholded mean test F1: `0.28129249401851025`
- Checkpoint SHA-256: `7ae7fd98d49ae16ae1d5811fb9a7f7a7c40ecfd1bbd0c3166ba584c38ec6fb53`

`test_results.json` contains the primary held-out test evaluation.
`thresholded_f1_results.json` records thresholds selected independently for
each pathology on the validation set and then applied unchanged to the test
set. This F1 analysis is secondary and does not reproduce the original
expert-labeled pneumonia comparison because that test set is not public.

`patient_splits.csv` is the exact patient assignment shared by the CheXNet and
DACNet runs. `threshold_run_environment.json` records the threshold-evaluation
environment and exact command. The checkpoint is distributed as a release
asset rather than tracked in Git history.
