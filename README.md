# MAGIC Gamma Telescope Binary Classification Pipeline

A supervised machine learning benchmark comparing **Gaussian Naive Bayes** and **$K$-Nearest Neighbors** on the MAGIC Gamma Telescope dataset, featuring data normalization and class-imbalance oversampling.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Dataset Characteristics](#dataset-characteristics)
- [Pipeline Architecture](#pipeline-architecture)
  - [1. Data Ingestion & Target Binarization](#1-data-ingestion--target-binarization)
  - [2. Train/Validation/Test Split](#2-trainvalidationtest-split)
  - [3. Feature Scaling & Oversampling](#3-feature-scaling--oversampling)
  - [4. Gaussian Naive Bayes Model](#4-gaussian-naive-bayes-model)
  - [5. K-Nearest Neighbors Model](#5-k-nearest-neighbors-model)
  - [6. Evaluation & Bug Analysis](#6-evaluation--bug-analysis)
- [Performance Summary](#performance-summary)

---

## 📖 Project Overview

This notebook implements a binary classification workflow to discriminate between primary gamma rays (`g`) and hadron showers (`h`) recorded by the MAGIC telescope. The workflow addresses class imbalance via random oversampling and tests probabilistic (`GaussianNB`) versus non-parametric (`KNeighborsClassifier`) decision boundaries.

---

## 📊 Dataset Characteristics

- **Instances**: 19,020
- **Features (10 continuous numerical attributes)**: `fLength`, `fWidth`, `fSize`, `fConc`, `fConc1`, `fAsym`, `fM3Long`, `fM3Trans`, `fAlpha`, `fDist`
- **Target (`class`)**: 
  - `g` (Gamma signal) $\rightarrow$ Encoded to `1`
  - `h` (Hadron background) $\rightarrow$ Encoded to `0`

---

## ⚙️ Environment Setup

Install necessary numerical, machine learning, and imbalance-handling dependencies:

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib
