# Methodology

## Problem formulation

Smartphone ISP maps sensor measurements to a display-referred image. Let `x` be
a RAW Bayer frame with CFA layout `C`, sensor black level `b`, white level `w`,
and camera metadata `m` (white balance, color matrix, lens shading, and so on).
The classical pipeline produces a linear RGB image and then a display image:

```text
I_linear = ISP_linear(x; C, b, w, m)
I_display = T(I_linear)
```

A learned restoration model `R_theta` can be inserted pre-ISP or post-ISP:

```text
PRE_ISP:     I = T(ISP_linear(R_theta(x)))
POST_ISP:    I = R_theta(T(ISP_linear(x)))
LEARNED_ONLY: I = R_theta(x)
```

The hybrid configuration is explicit. The model is not assumed to accept both
RAW and sRGB inputs with the same weights.

## Classical pipeline plan

### Bayer preprocessing

1. Estimate fixed or per-channel black level from masked/optically black
   regions when available.
2. Subtract the black level.
3. Normalize by `white_level - black_level`.
4. Clip to the configured range.
5. Preserve CFA and provenance metadata.

### White balance

Candidate estimators:

- Camera metadata gains (preferred when available).
- Gray-world statistics with robust trimming.
- White-patch / max-RGB with saturation rejection.
- Learned illuminant estimation as a later extension.

Gains are applied in the Bayer or RGB domain according to the configured stage
order. Gain limits prevent unrealistic corrections.

### Demosaicing

Candidate implementations:

- Bilinear interpolation as a reference.
- Malvar-He-Cutler gradient-corrected interpolation.
- Edge-aware directional interpolation.

Border handling and CFA phase alignment are part of the tests. The output is
linear RGB, never display sRGB.

### Color correction

Apply a 3x3 camera color matrix in linear RGB. The matrix can come from
metadata or configuration. Optional clipping is explicit. The working color
space is recorded on the output buffer.

### Denoising

Classical baselines:

- Bilateral filter.
- Non-local means.
- Guided filter.

The plan measures noise reduction against detail loss using both full-reference
metrics and targeted texture/noise fixtures. Denoising placement is explicit:
linear RGB before tone mapping by default.

### Tone mapping

Exposure adjustment followed by a selectable operator:

- Reinhard.
- Filmic curve.
- ACES approximation.
- Gamma.

The output is converted to sRGB with the documented transfer function. Tone
mapping parameters are recorded in evaluation reports.

### Sharpening

Unsharp masking with radius, amount, and threshold controls, plus an
edge-aware alternative. The plan includes halo and noise-amplification checks.

## Learned restoration model

The planned model is a residual U-Net:

- Configurable encoder depth and channel widths.
- Downsampling blocks with normalization and non-linear activation.
- Bottleneck with optional attention.
- Decoder with skip connections and bilinear or transposed-convolution
  upsampling.
- Residual output for reconstruction tasks.
- Optional dropout for regularization.

The first implementation targets either sRGB-to-sRGB restoration (SIDD-style)
or paired RAW/linear-to-sRGB restoration, not both with one checkpoint.

## Loss functions

Planned composite objective:

```text
L = w_charbonnier * L_charbonnier + w_ssim * L_ssim + w_lpips * L_lpips
```

- Charbonnier loss: robust pixel reconstruction.
- SSIM loss: structural similarity.
- LPIPS: optional perceptual term with a pinned backbone.

Weights are configuration values and must be reported. Loss terms are not
claimed to be implemented until their tests exist.

## Training plan

1. Deterministic seeds for Python, NumPy, and PyTorch.
2. Domain-aware paired dataset with scene-disjoint splits.
3. Synchronized crops and flips; no geometry mismatch between input and target.
4. Mixed precision on T4 where numerically stable.
5. Gradient clipping and validation at a fixed interval.
6. Checkpoint metadata includes config hash, git commit, dataset manifest hash,
   and environment versions.
7. Early stopping or fixed-budget training chosen before comparing variants.

Because the foundation has no training implementation, no training curves or
accuracy claims exist yet.

## Validation plan

Validation reports per-image and aggregate metrics:

- PSNR.
- SSIM.
- LPIPS.
- CIEDE2000 Delta E.
- Runtime and memory.

Aggregation must state whether values are means over images, crops, or scenes,
and whether any border cropping was applied. LPIPS and Delta E are distances
(lower is better); PSNR and SSIM are higher-is-better.

## Reproducibility checklist

- Git commit and branch.
- Model config and checkpoint hash.
- Dataset name, version, manifest hash, and split policy.
- CFA pattern, black/white levels, and color metadata for RAW data.
- Metric implementation and library versions.
- Random seeds.
- Hardware (Colab T4), CUDA, PyTorch, and driver versions.
- Warmup and timing protocol for efficiency numbers.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Treating sRGB as RAW | Explicit domain labels and runtime guards |
| Train/validation leakage | Scene-disjoint manifests and split checks |
| Metric overfitting | Report all four metrics and qualitative failures |
| Unfair ablations | Fixed seeds, budgets, and preprocessing |
| Hidden color conversions | Explicit stage order and recorded context |
| GPU session loss | Checkpoint download and Colab session logs |
| Unreproducible results | Pinned dependencies and recorded environment |

## Planned stages

The implementation sequence is documented in `docs/architecture.md`. Each stage
has a testable exit criterion and must not claim results from a later stage.
