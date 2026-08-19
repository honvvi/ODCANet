# ODCANet

Official PyTorch implementation of **Orthogonal Feature Decoupling and Hierarchical Cross-Scale Aggregation Network for RGB-T Semantic Segmentation**.

ODCANet consists of three components:
- **Shared Encoder** — weight-shared ConvNeXtV2-Base dual-stream encoder
- **OFDM** — Orthogonal Feature Decoupling Module (with CDB)
- **HCSAM Decoder** — Hierarchical Cross-Scale Aggregation Module

Supported benchmarks: **MFNet**, **PST900 (PST)**, and **FMB**.

## Table of Contents
1. [Requirements](#requirements)
3. [Dataset Preparation](#dataset-preparation)
4. [Checkpoints](#checkpoints)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Feature Decoupling Analysis](#feature-decoupling-analysis)
8. [Repository Layout](#repository-layout)

## Requirements
| Item | Version |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.10.0 |
| torchvision | 0.25.0 |
| CUDA | 12.x |

```bash
git clone https://github.com/honvvi/ODCANet.git
cd ODCANet
pip install -r requirements.txt
```

## Dataset Preparation
Set `--data_root` to the parent directory that contains the dataset folders.
`--dataset` must be one of `MFNet`, `PST`, or `FMB`, and must match the directory name under `--data_root`.

```text
<data_root>/                          # Parent path to --data_root
├── MFNet/                            # MFNet dataset 
│   ├── images/                       # 4-channel PNG (BGR + thermal)
│   │   └── <id>.png                  # Channels 0–2: RGB; channel 3: thermal
│   ├── labels/                       # Single-channel class-index masks
│   │   └── <id>.png                  # Same basename as the paired image
│   ├── train.txt                     # One sample id per line
│   └── test.txt                      # One sample id per line
├── PST/                              # PST900 dataset
│   ├── train/                        # Training split
│   │   ├── rgb/                      # RGB images
│   │   ├── thermal/                  # Thermal images
│   │   └── labels/                   # Single-channel class-index masks
│   └── test/                         # Test split
│       ├── rgb/                      # RGB images
│       ├── thermal/                  # Thermal images
│       └── labels/                   # Class-index masks
└── FMB/                              # FMB dataset
    ├── train/                        # Training split
    │   ├── Visible/                  # RGB images
    │   ├── Infrared/                 # Infrared images
    │   └── Label/                    # Single-channel class-index masks
    └── test/                         # Test split
        ├── Visible/                  # RGB images
        ├── Infrared/                 # Infrared images
        └── Label/                    # Class-index masks
```

Download links:
- **MFNet**: [Official site](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/)
- **PST900**: [Google Drive](https://drive.google.com/file/d/1hZeM-MvdUC_Btyok7mdF00RV-InbAadm/view?pli=1)
- **FMB**: [Baidu Pan](https://pan.baidu.com/s/1k7PgCsSJVZJIoIhgMjWxNg?pwd=IVIF#list/path=%2F)

Notes:
- **MFNet** uses the official four-channel PNG release. Place `images/`, `labels/`, `train.txt`, and `test.txt` under `MFNet/`. The loader reads each PNG as `[H, W, 4]`, takes channels 0–2 as RGB and channel 3 as thermal (repeated to 3 channels). No manual RGB/Thermal split is required. `train.txt` / `test.txt` list sample ids without the `.png` extension.
- Image resolutions used in our experiments: MFNet `640×480`, PST900 `1280×720`, FMB `800×600`.
## Checkpoints
Each dataset requires three component weights:

| File | Component |
| --- | --- |
| `shared_encoder.pth` | Shared Encoder |
| `ofdm.pth` | OFDM |
| `hcsam_decoder.pth` | HCSAM Decoder |

Place them under `Checkpoint/<dataset>/`:

```text
Checkpoint/
├── MFNet/
│   ├── shared_encoder.pth
│   ├── ofdm.pth
│   └── hcsam_decoder.pth
├── PST/
│   └── ...
└── FMB/
    └── ...
```

Checkpoint weight files are excluded from version control. Dataset-specific trained weights will be released on the repository **GitHub Releases** page. Pretrained ConvNeXtV2-Base (ImageNet-22K) is loaded automatically when `Model.Shared_Encoder.pretrain: True` in `config/train_cfg.yaml`.

## Training
```bash
python train.py \
  --config ./config/train_cfg.yaml \
  --dataset MFNet \
  --data_root /path/to/Datasets \
  --device cuda:0
```

Training applies ImageNet normalization to both RGB and thermal inputs, then the same Cv-style augment as `test/our` MotherData: random horizontal flip, multi-scale resize (`{0.5, 0.75, 1.0, 1.25, 1.5, 1.75}`), and crop/pad back to each image's original size (or to `--crop_height/--crop_width` when both are set). Checkpoints are selected by the best validation mIoU.

## Evaluation
```bash
python test.py \
  --config ./config/train_cfg.yaml \
  --dataset MFNet \
  --data_root /path/to/Datasets \
  --shared_encoder_path ./Checkpoint/MFNet/shared_encoder.pth \
  --ofdm_path ./Checkpoint/MFNet/ofdm.pth \
  --hcsam_decoder_path ./Checkpoint/MFNet/hcsam_decoder.pth \
  --device cuda:0
```

Replace `MFNet` with `PST` or `FMB` and update the checkpoint paths accordingly. Evaluation runs on CUDA and writes results to `./results/<dataset>/`, including `summary.json`, `efficiency_records.json`, and ROC-related outputs.


## Feature Decoupling Analysis
```bash
python utils/analyze_decoupling.py \
  --dataset MFNet \
  --data_root /path/to/Datasets \
  --config ./config/train_cfg.yaml \
  --device cuda:0 \
  --run "lambda=0:/path/shared_encoder.pth:/path/ofdm_lambda0.pth:/path/decoder.pth" \
  --run "lambda=0.01:/path/shared_encoder.pth:/path/ofdm_lambda001.pth:/path/decoder.pth" \
  --reference lambda=0 \
  --compute-cka
```

## Repository Layout
```text
ODCANet/
├── config/
│   └── train_cfg.yaml           # Training model hyperparameters (epochs, lr, orth_weight etc.)
├── data/
│   ├── __init__.py              # Unified build_data_loader / build_test_loader entry
│   ├── mfnet_dataset.py         # MFNet four-channel PNG loader and train/test split reader
│   └── rgbt_dataset.py          # PST / FMB RGB-T loaders, class metadata, and DataLoader builders
├── evaluation/
│   ├── __init__.py              # Evaluation package exports
│   ├── metrics.py               # mIoU, mAcc, pixel accuracy, per-class IoU/Acc
│   └── roc.py                   # ROC / AUC computation and related output writers
├── model/
│   ├── __init__.py              # Model package exports
│   ├── build_model.py           # Assembles Shared Encoder + OFDM + HCSAM into the full network
│   ├── shared_encoder.py        # Weight-shared ConvNeXtV2-Base dual-stream encoder
│   ├── ofdm.py                  # Orthogonal Feature Decoupling Module (OFDM) with CDB
│   └── hcsam.py                 # Hierarchical Cross-Scale Aggregation Module (HCSAM) decoder
├── utils/
│   ├── __init__.py              # Utils package exports
│   ├── analyze_decoupling.py    # CLI for feature-decoupling analysis
│   ├── feature_analysis.py      # Helpers used by decoupling analysis
│   ├── checkpoint.py            # Save / load component weights
│   ├── inference.py             # Forward-pass helpers for evaluation and timing
│   └── visualization.py         # Prediction / label colorization and figure helpers
├── Checkpoint/                  # Dataset-specific released weights
├── train.py                     # Training entry: load data, optimize, validate, save best mIoU weights
├── test.py                      # Evaluation entry: load checkpoints, report metrics / efficiency / ROC
├── requirements.txt             # Python dependencies
└── README.md                    
```

## Citation
If you find this repository useful, please cite the corresponding paper.
