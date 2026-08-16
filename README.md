# ODCANet

Code for **Orthogonal Feature Decoupling and Hierarchical Cross-Scale
Aggregation Network for RGB-T Semantic Segmentation**.

## Requirements

- Python 3.10
- PyTorch 2.10.0
- torchvision 0.25.0
- CUDA 12.x

Install dependencies:

```bash
pip install -r requirements.txt
```

## Repository Layout

```text
ODCANet/
├── Checkpoint/
│   ├── .gitignore               # Prevent checkpoint binaries from being committed
│   └── README.md                # Place dataset-specific component weights here
├── config/
│   └── train_cfg.yaml
├── data/
│   ├── __init__.py
│   └── rgbt_dataset.py          # Dataset metadata, image loading, DataLoader
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py               # mIoU, mAccuracy, pixel accuracy, per-class IoU
│   ├── roc.py                   # ROC/AUC collection and output
├── model/
│   ├── __init__.py
│   ├── build_model.py
│   ├── shared_encoder.py         # Shared ConvNeXtV2 encoder
│   ├── ofdm.py                   # OFDM and OPB
│   └── hcsam.py                  # HCSAM and PUB
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py             # Checkpoint compatibility and loading
│   ├── inference.py              # Model forward helper
│   └── visualization.py          # Prediction visualization
├── test.py                       # Evaluation entry point
├── requirements.txt
└── README.md
```

`test.py` contains the evaluation workflow, including model construction,
validation, result printing, and optional ROC output. Dataset loading,
checkpoint conversion, model inference, and metric implementations are kept
in their own modules.

## Dataset Layout

The `--dataset` value and its directory name must be one of `FMB`, `PST`, or
`MH`. Use the following layout below `--data_root`:

```text
Datasets/
├── FMB/
│   └── test/
│       ├── Visible/
│       ├── Infrared/
│       └── Label/
├── PST/
│   └── test/
│       ├── rgb/
│       ├── thermal/
│       └── labels/
└── MH/
    └── test/
        ├── <id>_rgb.png
        ├── <id>_th.png
        └── <id>.png
```

For FMB and PST, each RGB image, thermal image, and label must share the same
filename. MH uses the `<id>_rgb.png`, `<id>_th.png`, and `<id>.png` naming
convention used by the original experiment code.

## Checkpoints

Supply one checkpoint for each component:

- `Shared_Encoder`
- `OFDM`
- `HCSAM_Decoder`

Store them under `Checkpoint/<dataset>/`, for example:

```text
Checkpoint/FMB/
├── shared_encoder.pth
├── ofdm.pth
└── hcsam_decoder.pth
```

The loader validates tensor names and shapes. It also converts checkpoints
saved by the former training code using `Teacher_Encoder`, `Teacher_Fusion`,
and `Teacher_Decoder` names.

## Evaluation

Each invocation performs one complete CUDA evaluation on `cuda:2`: mIoU,
mAccuracy, pixel accuracy, per-class IoU, Params, FLOPs, peak memory,
latency, FPS, pure inference time, test runtime, and ROC/AUC.

```bash
python3 test.py \
  --config ./config/train_cfg.yaml \
  --dataset FMB \
  --data_root /path/to/Datasets \
  --shared_encoder_path ./Checkpoint/FMB/shared_encoder.pth \
  --ofdm_path ./Checkpoint/FMB/ofdm.pth \
  --hcsam_decoder_path ./Checkpoint/FMB/hcsam_decoder.pth
```

The former option names `--encoder_path`, `--fusion_path`, and
`--decoder_path` are retained for compatibility.

Results are written automatically to `results/FMB/`:

- `roc_curve.png`
- `roc_curves.npz`
- `summary.json`
- `efficiency_records.json`
