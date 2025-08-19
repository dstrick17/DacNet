---
title: 'An Open-Source Reproduction and Enhancement of CheXNet for Chest X-ray Disease Classification'
tags:
  - Python
  - Deep Learnings
  - PyTorch
  - Medical Imaging
  - Computer Vision
  - Radiology
  - NIH ChestX-ray14
  - Disease Classification
authors:
  - name: Daniel J. Strick
    orcid: 0009-0000-1715-8187
    corresponding: true
    affiliation: 1
  - name: Carlos Fernando Garcia Padilla
    equal-contrib: true
    affiliation: 1
  - name: Anthony Tse Huang

# Summary
This project aimed to rigorously reproduce and extend CheXNet, a landmark deep learning
model in medical imaging. Through our efforts, we demonstrated that meaningful improvements
to the original architecture can be achieved by incorporating techniques developed
since the paper’s publication. Specifically, we found that the use of Focal Loss, the AdamW
optimizer with weight decay, Color Jitter for data augmentation, ReduceLROnPlateau
scheduling, and per-disease F1 threshold tuning substantially improved model stability and
performance across all 14 thoracic disease classes. Our model, "DannyNet," achieved a
strong balance between interpretability and predictive power, reaching an average AUC of
0.85 and an F1 score of 0.39. These results highlight the potential for targeted enhancements
to significantly improve performance on imbalanced and clinically relevant datasets like NIH
ChestX-ray14. Beyond the technical contributions, this project underscores the importance
of reproducibility in machine learning research. We were motivated not only to validate the
claims of a high-impact study, but also to contribute openly to the scientific community.
Our full codebase, including model configurations, evaluation tools, and visualization scripts,
has been made publicly available in hopes of promoting transparency and enabling further
development by others. Ultimately, this work contributes to both the biomedical and data science communities by showing how modern training strategies can elevate model
performance in critical clinical applications. We hope our findings serve as a foundation for
future work exploring deep learning in medical diagnostics and for building more accurate,
interpretable, and equitable AI tools in healthcare.

# Statement of need
Chest X-ray classification is a crucial task in medical image analysis, where deep learning
models are trained to detect various thoracic diseases from radiographic scans. A landmark
study in this field, known as CheXNet, introduced a 121-layer DenseNet convolutional neural
network that reportedly outperformed radiologists in detecting pneumonia [1]. Their work
used the NIH ChestX-ray14 dataset, a publicly available dataset of over 100,000 frontal-view
chest X-rays labeled with up to 14 disease classes [2]. The success of CheXNet has inspired
further research, as it represents a significant step toward using artificial intelligence to assist
in clinical diagnosis, especially in regions where access to licensed radiologists is limited [3].
In the midst of a reproducibility crisis in academia, independent researchers must reproduce
groundbreaking studies like this in order to help guide future research [4]. In this project, we
set out to replicate the original CheXNet model as closely as possible, evaluate and improve
performance metrics such as AUC-ROC and F1 scores across all 14 disease classes, and
explore whether newer deep learning techniques, particularly Vision Transformers (ViTs),
could offer performance improvements over traditional convolutional neural networks. All
code for our models and evaluation pipeline is publicly available in our GitHub Repository
- https://github.com/dstrick17/DannyNet. While our primary goal was to
replicate the original CheXNet study, we also recognize the importance of its successor,
CheXNeXt, which validated a similar model against board-certified radiologists on a curated
internal dataset [5]. Although the test set used in CheXNeXt is not publicly available,
its findings emphasize the clinical relevance of these models and reinforce the need for
reproducible benchmarking in public datasets like NIH ChestX-ray14.
Our key contributions are as follows: We performed a faithful replication of the CheXNet
model, establishing a reproducible baseline using pretrained DenseNet-121 with standard
training procedures. We proposed an improved model, DannyNet, which incorporates Focal Loss, the AdamWoptimizer, and advanced image augmentations like Color Jitter. It achieved
significantly higher F1 scores on rare classes compared to the baseline. We implemented
per-class F1 threshold optimization to further boost classification accuracy, especially in
multi-label settings. Unlike the original CheXNet study, which only reported an F1 score for
pneumonia using a non-public expert-labeled subset, our study computes per-class F1 scores
across all 14 diseases using a reproducible patient-wise split. This provides a more granular
view of model strengths and limitations in multi-label medical image classification. We
explored the use of transformer-based models (ViT) for X-ray classification, benchmarking
their performance against CNN-based architectures. Finally, we developed a Streamlit web
app hosted on Hugging Face that takes a chest X-ray input, returns disease predictions
using DannyNet, and overlays Grad-CAM heatmaps to visualize model attention.

# Methods
We built on DenseNet-121 but replaced the
BCE loss with Focal Loss (gamma=2, alpha=1) to address extreme class imbalance.
We used the AdamW optimizer with weight decay, a learning rate of 0.00005, and a
ReduceLROnPlateau scheduler. Augmentations included RandomResizedCrop(224),
RandomHorizontalFlip, and ColorJitter. This model was trained using a patient-level
split and achieved a test AUC of 0.85, a test loss of 0.04, and an average F1 score of 0.39.
We believe the focal loss contributed significantly to reducing test loss and improving
prediction confidence on minority classes. This model outperformed CheXNet in AUC
for 9 out of 14 diseases.

# Citations


# Figures

# Acknowledgements

# References