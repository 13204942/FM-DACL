# FM-DACL: Dual Agreement Consistency Learning with Foundation Models  
### ISBI 2026 FETUS Challenge Submission

This repository contains the official implementation of **FM-DACL**, a semi-supervised learning framework for fetal heart ultrasound segmentation and diagnosis developed for the **[ISBI 2026 FETUS Challenge (Fetal HearT UltraSound Segmentation and Diagnosis)](http://119.29.231.17:90/)**.

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

## Installation

### Requirements

Recommended environment:
```
albumentations==2.0.5
h5py==3.12.1
matplotlib==3.10.8
MedPy==0.5.2
monai==1.5.2
numpy==2.4.3
opencv_python==4.10.0.84
opencv_python_headless==4.11.0.86
Pillow==12.1.1
scikit_learn==1.8.0
scipy==1.17.1
segmentation_models_pytorch==0.4.0
skimage==0.0
tensorboardX==2.6.2.2
tensorboardX==2.6.4
torch==2.5.1
torchvision==0.20.1
tqdm==4.67.1
```

## Dataset Preparation

Prepare the dataset following the FETUS challenge format.
```
FETUS2026/
│
├── 📄 train_labeled.json      # 🏷️ Labeled training config
├── 📄 train_unlabeled.json    # 🔄 Unlabeled training config
├── 📄 valid.json             # ✅ Validation config
├── 🖼️ images/                # Medical ultrasound images
│   ├── 1699.h5
│   └── ... (thousands of cases)
└── 🏷️ labels/                # Ground truth annotations
    ├── 1699_label.h5
    └── ... (matching labels)
```
Update dataset paths in training scripts accordingly.

**Image Files (images/*.h5)**
```
{
    'image': np.ndarray,  # (512, 512, 3) uint8 RGB ultrasound image
    'view': np.ndarray    # (1,) int32 view ID (1: 4CH, 2: LVOT, 3: RVOT, 4: 3VT)
}
```
**Label Files (labels/*_label.h5)**
```
{
    'mask': np.ndarray,   # (512, 512) uint8 segmentation mask (0-14 classes)
    'label': np.ndarray   # (7,) uint8 binary classification labels
}
```

## Training
Example training commands:

### Consistency training
```
python train_semi_echocare_unet_consist.py --root_path /root/FM-DACL/log --exp EchoCare_Unet --max_iterations 78200
```

### CPS training
```
python train_semi_echocare_unet_cps.py --root_path /root/FM-DACL/log --exp EchoCare_Unet_CPS --max_iterations 78200
```

More examples are provided in:
```
runner.txt
```

## Inference
Example inference command:
```
python step_2_inference.py \
    --data-json /root/FM-DACL/data/validation.json \
    --ckpt /root/FM-DACL/log/EchoCare_resunet_consist_391/best_model1.pth \
    --out-dir /root/FM-DACL/predictions \
    --mask-mode oracle \
    --cls-thr 0.5 \
    --gpu 0
```
Output:
```
🎯 ./predictions/
├── 📄 1.h5    # ❤️ Fetal cardiac analysis
├── 📄 2.h5    # 🏥 Segmentation & classification
└── 📄 ... (all test cases)

📊 Each H5 file contains:
├── 🖼️ mask:  Segmentation predictions (H×W)
└── 🏷️ label: Classification results (7 classes)
```
## Evaluation
Performance Analysis
```
python step_3_evaluate.py \
    --valid-json data/valid.json \
    --pred-dir ./predictions \
    --save-dir ./evaluation_results
```

## Key Features

FM-DACL introduces:

- Dual agreement consistency learning
- Heterogeneous model collaboration
- Semi-supervised training with unlabeled data
- Foundation model integration
- Robust multi-task learning framework

## Citation

If you use this work, please cite:


## License

MIT License.

## Contact

Fangyijie Wang  
PhD Researcher  
University College Dublin  

For questions, please open an issue.