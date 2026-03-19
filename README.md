# FM-DACL: Dual Agreement Consistency Learning with Foundation Models  
### ISBI 2026 FETUS Challenge Submission

This repository contains the official implementation of **FM-DACL**, a semi-supervised learning framework for fetal heart ultrasound segmentation and diagnosis developed for the **ISBI 2026 FETUS Challenge (Fetal HearT UltraSound Segmentation and Diagnosis)**.

## Overview

Accurate fetal heart ultrasound analysis is challenging due to limited annotated data and large anatomical variability. To address this problem, we propose **FM-DACL**, a semi-supervised framework that combines:

- Foundation model representations
- Heterogeneous model collaboration
- Dual agreement consistency learning
- Cross pseudo supervision
- Interpolation consistency regularization

The framework leverages both labeled and unlabeled data to improve robustness and generalization.

## Method Pipeline

FM-DACL consists of two main stages:

### Stage 1 — Semi-supervised Training
Two heterogeneous networks are jointly trained:

Examples include:
- Swin-UNet
- SegFormer + ResUNet
- EchoCare backbone variants

Training strategy includes:

- Cross pseudo supervision (CPS)
- Dual agreement consistency loss
- Entropy regularization
- Consistency learning on unlabeled data

### Stage 2 — Inference
The trained model is used for:

- Segmentation prediction
- Classification prediction
- Mask refinement
- Threshold-based diagnosis

## Repository Structure
