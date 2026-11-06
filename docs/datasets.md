# Dataset Preparation

## Non-negotiable rule

SIDD images distributed as sRGB PNGs are **display-referred RGB images, not
Bayer RAW mosaics**. They must be labeled `srgb`. The repository rejects a
configuration that declares a `.png`/`.jpg` input as `bayer_raw`
(`DomainMismatchError`).

If a RAW dataset is used, its CFA pattern, bit depth, black level, white level,
white-balance metadata, and color matrix must be documented in the manifest and
validated before training.

## Supported domains

| Domain | Typical files | Meaning |
| --- | --- | --- |
| `bayer_raw` | `.npy`, `.raw`, `.tif`, `.tiff`, `.dng`, `.arw`, `.cr2`, `.nef` | Single-channel sensor mosaic |
| `linear_rgb` | `.npy`, `.tif`, `.tiff`, `.exr` | Scene-linear RGB |
| `srgb` | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`, `.tif`, `.tiff` | Display-referred RGB |

Extensions alone do not prove a domain; the manifest declaration is primary and
the extension check is a safety net.

## Repository layout for a local dataset

Datasets are not committed. A typical local layout is:

```text
data/
└── sidd-srgb/
    ├── train/
    │   ├── input/
    │   │   ├── scene_0001.png
    │   │   └── ...
    │   └── target/
    │       ├── scene_0001.png
    │       └── ...
    ├── val/
    │   ├── input/
    │   └── target/
    └── manifest.yaml
```

Input and target files are paired by matching relative stem. Nested directories
are supported. Unmatched files cause an error.

For a RAW-to-sRGB dataset, input and target may use different extensions:

```text
data/raw-pairs/
└── train/
    ├── input/
    │   └── scene_0001.npy
    └── target/
        └── scene_0001.png
```

## Manifest schema

`configs/datasets/` contains example manifests. A SIDD-style sRGB manifest:

```yaml
dataset:
  name: SIDD
  domain: srgb
  display_referred: true
  input_dir: input
  target_dir: target
  input_extensions: [.png]
  target_extensions: [.png]
  input_domain: srgb
  target_domain: srgb
  crop_size: 256
  split_policy: scene_disjoint
  notes: "SIDD sRGB PNGs are display-referred, not Bayer RAW."
```

A RAW manifest must add:

```yaml
dataset:
  domain: bayer_raw
  cfa: RGGB
  sensor:
    bit_depth: 16
    black_level: 64
    white_level: 4095
```

Do not guess these values. Read them from the dataset documentation or camera
metadata. Incorrect values invalidate both training and evaluation.

## Split policy

- Split by scene, not by crop. Crops from the same scene must not appear in both
  training and validation.
- Keep the official test split untouched until final evaluation.
- Record the manifest hash and split assignment for every run.
- Do not tune hyperparameters on the test split.

The planned dataset adapter will reject duplicate scene identifiers across
splits once scene metadata is present. Until then, split discipline is a manual
review step.

## Preprocessing plan

Stage 2 will implement:

1. Domain-aware decoding.
2. Black-level subtraction and normalization for RAW data.
3. Explicit sRGB-to-linear conversion only when a model is trained in linear
   space.
4. Synchronized random crops and geometry-only augmentation.
5. Deterministic validation preprocessing.
6. Tensor conversion with documented channel order and value range.

No preprocessing step may silently convert an sRGB image into Bayer RAW.

## Checksums and provenance

Each prepared dataset should include:

- Source URL or local origin.
- Version or download date.
- File count and checksum of the manifest.
- Split policy.
- Known exclusions or corrupt files.
- License or usage constraints.

The training configuration and report should record the manifest hash.

## What stays out of Git

- `data/`, `datasets/`.
- Checkpoints (`*.pth`, `*.pt`, `*.ckpt`, `*.safetensors`, `checkpoints/`).
- Generated outputs and logs.
- Credentials, `.env`, and `.streamlit/secrets.toml`.

The `.gitignore` enforces these exclusions. Never commit a dataset URL that
contains a token or signed credential.

## Planned commands

The dataset adapter is an interface in this release. Once implemented, the
intended workflow is:

```bash
uv run python -m smartisp.data.prepare \
  --manifest configs/datasets/sidd_srgb.yaml \
  --output data/sidd-srgb

uv run python -m smartisp.training.train \
  --config configs/default.yaml \
  --override configs/datasets/sidd_srgb.local.yaml \
  --dry-run
```

The `prepare` module does not exist yet; it is intentionally not stubbed as a
working command.
