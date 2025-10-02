**Performance vs older models and publications on Test AUC scores per disease**
| Pathology           | original CheXNet | Dacnet.py | vit_transformer.py | replicate_chexnet.py |
|---------------------|------------------|----------|-------------------|--------------------|
| Atelectasis         | 0.8094           | **0.817** | 0.774           | 0.762              |
| Cardiomegaly        | 0.9248           | **0.932** | 0.89            | 0.922              |
| Consolidation       | **0.7901**       | 0.783     | 0.789           | 0.746              |
| Edema               | 0.8878           | **0.896** | 0.876           | 0.864              |
| Effusion            | 0.8638           | **0.905** | 0.857           | 0.883              |
| Emphysema           | 0.9371           | **0.963** | 0.828           | 0.85               |
| Fibrosis            | 0.8047           | **0.814** | 0.772           | 0.766              |
| Hernia              | 0.9164           | **0.997** | 0.872           | 0.925              |
| Infiltration        | **0.7345**       | 0.708     | 0.7             | 0.673              |
| Mass                | 0.8676           | **0.919** | 0.783           | 0.824              |
| Nodule              | 0.7802           | **0.789** | 0.673           | 0.646              |
| Pleural Thickening  | **0.8062**       | 0.801     | 0.766           | 0.756              |
| Pneumonia           | **0.768**        | 0.74      | 0.713           | 0.656              |
| Pneumothorax        | **0.8887**       | 0.875     | 0.821           | 0.827              |