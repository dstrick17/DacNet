Optimal pneumonia threshold about 0.12

- consider combining with kaggle pneumonia dataset

- Consider adding focul loss to focus on rare diseases , can then add deeper focus loss
- Consider adding patient demographics


# Similar paper in nanture: https://www.nature.com/articles/s41551-022-00936-9

CONFIG["batch_size"] = 64  # or 128 if your GPU has >10GB VRAM
CONFIG["learning_rate"] = 0.0005  # roughly scaled up
CONFIG["num_workers"] = 8  # if you're not already at 8


## 📊 Experiment Leaderboard

| Run Name           | Git Commit | Strategy              | Pneumonia F1 | Avg AUC | Val F1 | Notes                        | WandB Link |
|--------------------|------------|-----------------------|--------------|---------|--------|------------------------------|------------|
| baseline_v1        | 3b2f1e1    | no oversampling        | 0.00         | 0.72    | 0.01   | Original setup               | [link](https://wandb.ai/...) |
| oversample_pneu    | 7c8e4a2    | pneumonia oversampled  | **0.34**     | 0.70    | 0.05   | Balanced train set           | [link].) |

