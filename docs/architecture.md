# SmartISP Architecture

## Goals

SmartISP provides a modular, testable foundation for hybrid smartphone ISP
research. The architecture prioritizes:

1. Explicit data domains and provenance.
2. Separable classical stages with stable contracts.
3. A learned restoration model that can be inserted at documented points.
4. Reproducible training and evaluation on Google Colab T4.
5. Honest reporting: interfaces are allowed to be incomplete, results are not
   invented.

Non-goals for the foundation release are trained models, benchmark claims, and
production camera integration.

## Layer overview

```text
+-----------------------+
| Configuration (YAML)  |
+-----------+-----------+
            |
+-----------v-----------+
| Data + provenance     |  domain labels, pair discovery, splits
+-----------+-----------+
            |
+-----------v-----------+
| Hybrid pipeline       |  CLASSICAL_ONLY / LEARNED_ONLY / PRE_ISP / POST_ISP
+-----+-----------+-----+
      |           |
+-----v-----+ +---v------------------+
| Classical | | Learned restoration  |
| ISP       | | U-Net + losses       |
+-----+-----+ +---+------------------+
      |           |
+-----v-----------v-----+
| Evaluation            |  PSNR, SSIM, LPIPS, Delta E, latency, memory
+-----------------------+ 
```

## Data model

Every buffer carries its domain.

| Type | Shape | Meaning |
| --- | --- | --- |
| `RawFrame` | `(H, W)` | Single-channel Bayer mosaic with CFA and sensor levels |
| `RGBImage` | `(H, W, 3)` | Three-channel image with an explicit `ColorSpace` |
| `ISPContext` | metadata | CFA, black/white level, camera WB, CCM, working space |

`Domain` is a separate label for files:

- `bayer_raw`: sensor mosaic.
- `linear_rgb`: scene-linear RGB.
- `srgb`: display-referred sRGB.

The guard in `smartisp.data.domains` rejects obviously contradictory
combinations, for example a `bayer_raw` declaration for a `.png` file. This is
the code-level enforcement of the rule that SIDD sRGB images are not Bayer RAW.

## Classical ISP

The default stage order is:

```text
Bayer preprocess -> white balance -> demosaic -> color correction
                 -> denoise -> tone mapping -> sharpen
```

Rationale:

- Black-level correction and normalization happen in the Bayer domain.
- White balance can run before demosaicing when camera metadata or CFA-aware
  statistics are available.
- Demosaicing produces linear RGB with a documented CFA-aware border policy.
- Color correction maps camera primaries into the working RGB space.
- Denoising operates in linear RGB before the tone curve can amplify noise.
- Tone mapping produces a display-referred image.
- Sharpening is last so it can account for the final output encoding.

Each stage implements the `ISPStage` protocol:

```python
class ISPStage(Protocol):
    name: str

    def process(self, image: Any, context: ISPContext | None = None) -> Any: ...
```

The protocol is deliberately loose at the type boundary because stages consume
and produce different domains. Concrete signatures document the intended input
and output types:

| Stage | Input | Output | Planned implementation |
| --- | --- | --- | --- |
| `BayerPreprocessor` | `RawFrame` | `RawFrame` | Black-level estimation, normalization, clipping |
| `WhiteBalancer` | `RawFrame` or `RGBImage` | same | Metadata, gray-world, white-patch estimators |
| `Demosaicer` | `RawFrame` | `RGBImage` | Bilinear, Malvar, edge-aware |
| `ColorCorrector` | `RGBImage` | `RGBImage` | 3x3 CCM in linear RGB |
| `Denoiser` | `RGBImage` | `RGBImage` | Bilateral, non-local means, guided |
| `ToneMapper` | `RGBImage` | `RGBImage` | Reinhard, filmic, ACES, gamma |
| `Sharpener` | `RGBImage` | `RGBImage` | Unsharp mask, Laplacian, guided |

All stages are current interface placeholders and raise `NotImplementedError`
with a description of the planned behavior.

## Learned restoration

`UNetRestorer` fixes the model contract:

- Input tensor: `(N, C_in, H, W)`.
- Output tensor: `(N, C_out, H, W)`.
- Configuration: channel widths, depth, normalization, activation, dropout,
  residual prediction, optional attention, and upsampling mode.
- `restore()` is the hybrid-pipeline entry point.

The planned encoder/decoder uses skip connections and optional residual output
(`out = input + residual`). Architecture construction, loss terms, and training
are stage 2 and stage 3 work. The model is intentionally not faked.

Loss configuration supports:

- Charbonnier / L1 reconstruction.
- SSIM structural loss.
- Optional LPIPS perceptual loss.
- Weighted composite objective.

## Hybrid placement

```text
PRE_ISP:    RAW/linear -> restoration -> classical ISP -> output
POST_ISP:   RAW/linear -> classical ISP -> restoration -> output
LEARNED_ONLY: input -> restoration -> output
CLASSICAL_ONLY: input -> classical ISP -> output
```

`PRE_ISP` and `POST_ISP` require different checkpoints and input domains. The
hybrid pipeline validates that the required components exist, but it does not
silently convert color spaces or invent missing metadata. Domain compatibility
checks are a stage 1-2 addition.

## Package boundaries

| Package | Responsibility | Must not do |
| --- | --- | --- |
| `smartisp.types` | Shared buffers and metadata | Import torch or OpenCV |
| `smartisp.config` | YAML loading and environment expansion | Contain dataset-specific logic |
| `smartisp.classical` | Classical stage contracts and composition | Load checkpoints |
| `smartisp.data` | Domain guards, pair discovery, transforms | Assume sRGB is RAW |
| `smartisp.models` | Model and loss contracts | Import at package import time |
| `smartisp.pipeline` | Hybrid composition | Perform hidden conversions |
| `smartisp.training` | Training/validation entry points | Hard-code local paths |
| `smartisp.evaluation` | Metrics, efficiency, analysis | Fabricate results |
| `apps` | Demonstration UI | Define core behavior |
| `scripts` | GPU and workflow helpers | Store credentials |

Torch-dependent modules are imported lazily from `smartisp.models.__init__` and
inside the data loader builder. This keeps configuration, dataset discovery,
and interface tests usable in a lightweight CPU environment.

## Training data flow

```text
Manifest + split
      |
      v
PairedDatasetConfig --(domain guards)--> PairedImageDataset
      |
      v
decode -> normalize -> synchronized crop/augment -> tensors
      |
      v
DataLoader -> U-Net -> loss -> optimizer -> checkpoint
      |
      v
Validation metrics + efficiency report
```

The foundation implements manifest parsing, pair discovery, and domain guards.
Decoding, normalization, augmentation, and tensor conversion are stage 2.

## Failure handling

- Interface placeholders raise `NotImplementedError` with the planned behavior.
- Domain contradictions raise `DomainMismatchError`.
- Missing or unmatched files raise `FileNotFoundError` or `ValueError`.
- The CLIs support `--dry-run` so configuration can be validated before
  training exists.

## Planned implementation stages

| Stage | Modules | Deliverable |
| --- | --- | --- |
| 0 | All scaffolds | Importable package, tests, docs, Colab workflow |
| 1 | `classical.*` | Real algorithms, deterministic fixtures, artifact tests |
| 2 | `data.*`, `models.*` | Loaders, model forward, losses, unit tests |
| 3 | `training.*`, `scripts.*` | Seeded T4 training, checkpoints, validation |
| 4 | `evaluation.*` | Real metric, latency, ablation, failure reports |
| 5 | `apps.*` | Demonstration app backed by available checkpoints |

Stage gates are explicit: no stage may claim results from a later stage.
