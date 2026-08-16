# Checkpoint Directory

Place the three component checkpoints for each dataset in a subdirectory.

```text
Checkpoint/
└── FMB/
    ├── shared_encoder.pth
    ├── ofdm.pth
    └── hcsam_decoder.pth
```

Use the same layout for `PST` and `MH`. Checkpoint files are intentionally
excluded from version control.
