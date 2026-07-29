# Publication Checklist for ReScience C

This checklist is scoped to preparing DACNet for a ReScience C submission.

## ReScience C Requirements

Source: https://rescience.github.io/write/

- Public code repository: complete.
- Public data repository or clear public data access instructions: mostly complete. The README and `DATA_ACCESS.md` point to Kaggle and NIH/Box access paths; add any generated splits, outputs, or non-Kaggle artifacts needed for exact verification.
- Article source and metadata: missing. Use the ReScience C LaTeX/template metadata before submission.
- Reviewable reproduction commands: complete for the three training scripts via `--data_dir` or `NIH_DATA_DIR`.
- Evidence for replicated results: partially complete. Tables, figures, and WandB metadata are present; exact generated result JSON files should be archived from the final reviewed run.
- Code DOI after acceptance: missing. Archive the accepted release on Zenodo.
- Scope check: confirm this is a replication of work by authors who are not the repository authors or close collaborators.

## Reviewer Run Path

```bash
export NIH_DATA_DIR=/path/to/nih_data
bash reproduce.sh
python scripts/replicate_chexnet.py --data_dir "$NIH_DATA_DIR" --wandb_mode offline
python scripts/dacnet.py --data_dir "$NIH_DATA_DIR" --wandb_mode offline
python scripts/vit_transformer.py --data_dir "$NIH_DATA_DIR" --wandb_mode offline
```

Each training script writes `models/<run_id>/test_results.json`.

## Repository Cleanup Before Submission

- Tracked `.DS_Store` files have been removed.
- The duplicate draft PDF workflow has been removed.
- Add final article source under `paper/` or the journal template's expected path.
- Add citation metadata once the final author list, DOI, and preferred citation are fixed.
- Verify a fresh clone can run `bash reproduce.sh`.
- Verify a final GPU run regenerates the reported tables.
