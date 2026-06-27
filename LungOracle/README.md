---
title: LungOracle
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: true
license: mit
---

# LungOracle — LUAD Survival Prediction

**AI-powered personalised survival risk prediction for Lung Adenocarcinoma**

LungOracle integrates gene expression, tumour dysregulation, immune microenvironment, and clinical variables into an ensemble machine learning framework for personalised survival prediction.

## Model Performance

| Metric | Value |
|--------|-------|
| C-index (TCGA-LUAD, n=478) | 0.702 |
| Stage I C-index | 0.888 vs 0.619 (clinical baseline) |
| External validation (GSE72094) | 0.636 |
| External validation (GSE68465) | 0.637 |

## Features

- **124 molecular and clinical features**
  - 72 Lasso-selected expression genes
  - 20 tumour dysregulation z-scores (vs GTEx healthy tissue)
  - 22 immune cell fractions (DIY CIBERSORT)
  - 5 clinical features (age, sex, stage)
  - 5 interaction terms

- **Two modes**
  - Demo Mode: 10 pre-selected TCGA patients with instant results
  - Predict Mode: Upload your own RNA-seq gene expression file

- **Full explainability**
  - Kaplan-Meier survival curve with patient position
  - SHAP feature importance
  - Immune microenvironment profile

## Research Paper

> Beyond the Tumor: Integrating Transcriptomic Dysregulation and Immune Signatures for Survival Prediction in Lung Adenocarcinoma

Supervised by Professor Alessandro Vinciarelli, University of Glasgow.

## Disclaimer

LungOracle is a research tool only. It is not approved for clinical diagnosis or treatment decisions. Always consult a qualified healthcare professional.

## GitHub

[ParthShringarpure09/Lung_adenocarcinoma_survival](https://github.com/ParthShringarpure09/Lung_adenocarcinoma_survival)
