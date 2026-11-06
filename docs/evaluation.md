# Evaluation Procedure

## Status

Metric functions and efficiency measurement are interfaces in this release.
They raise `NotImplementedError` and produce no numbers. The ablation matrix and
failure taxonomy are plans, not completed analyses.

## Image-quality metrics

### PSNR

Peak signal-to-noise ratio for prediction `y_hat` and reference `y`:

```text
PSNR = 10 * log10(peak^2 / MSE)
```

Report:

- Data range and channel handling.
- Per-image values and the aggregation rule (mean over images, not over pixels).
- Any border crop used to avoid demosaic boundary effects.

### SSIM

Structural similarity with a documented window size, Gaussian sigma, and
channel reduction. Report the implementation source and parameters. A common
protocol uses an 11x11 Gaussian window with sigma 1.5.

### LPIPS

Learned perceptual similarity. LPIPS is reported as a **distance** (lower is
better). Pin the backbone (`alex`, `vgg`, or `squeeze`), pretrained weight
source, input normalization, and library version. Pretrained weights must be
downloaded in the remote environment and are never committed.

### Delta E

CIEDE2000 color difference:

1. Convert sRGB predictions and references to linear RGB with the documented
   transfer function.
2. Convert to XYZ using the D65 white point.
3. Convert to CIELAB.
4. Compute CIEDE2000 per pixel.
5. Report the mean and distribution.

Do not compare Delta E values computed under different white points or color
pipelines.

## Metric direction summary

| Metric | Direction |
| --- | --- |
| PSNR | Higher is better |
| SSIM | Higher is better |
| LPIPS | Lower is better |
| CIEDE2000 Delta E | Lower is better |

## Runtime and memory measurements

Measurement protocol:

1. Move the model and a fixed input to the target device.
2. Run warmup iterations (`EfficiencyConfig.warmup_runs`).
3. Synchronize the device.
4. Measure each timed run (`EfficiencyConfig.timed_runs`) with a high-resolution
   timer.
5. Synchronize after each batch when measuring GPU latency.
6. Report p50, p90, p95, mean latency, and throughput.
7. Report peak GPU memory with `torch.cuda.max_memory_allocated` and host RSS
   with `psutil`.

Required metadata:

- Device name and compute capability.
- Input shape, dtype, and batch size.
- PyTorch, CUDA, and driver versions.
- Mixed-precision setting.
- Whether preprocessing and host-to-device transfer are included.
- Warmup count and timed run count.

Runtime numbers are not comparable across different input resolutions, batch
sizes, dtypes, or synchronization policies.

## Ablation plan

`default_ablation_plan()` defines the planned variants:

| Variant | Purpose |
| --- | --- |
| `classical_baseline` | Non-learned reference |
| `learned_only` | Learned capacity without classical priors |
| `hybrid_pre_isp` | Restoration in the RAW/linear domain |
| `hybrid_post_isp` | Restoration on display-referred output |
| `no_denoise` | Contribution of classical denoising |
| `no_tone_mapping` | Tone-curve effect on perceptual metrics |
| `no_sharpen` | Halos and noise amplification from sharpening |

Rules:

- Fix seeds, data splits, training budget, and validation protocol.
- Report all four image-quality metrics for each variant.
- Include efficiency measurements for the same hardware and input shape.
- Do not claim a variant is better without both metric and visual evidence.

## Failure-case analysis

Planned categories:

- Highlight clipping.
- Low-light noise.
- Color cast.
- Demosaic artifacts.
- Moiré.
- Motion blur.
- Domain mismatch.
- Learned artifacts such as checkerboarding or texture hallucination.
- Other.

A failure record includes:

```yaml
case_id: unique-id
category: low_light_noise
description: short reproducible observation
metrics:
  psnr: null
  ssim: null
  lpips: null
  delta_e_2000: null
artifacts:
  - outputs/failure-cases/example/input.png
  - outputs/failure-cases/example/prediction.png
  - outputs/failure-cases/example/reference.png
```

Metrics are filled only from real predictions. Failure categories are assigned
after visual inspection and, where applicable, diagnostic measurements.

## Reporting

A reproducible evaluation report should include:

- Git commit, branch, and dirty-worktree status.
- Configuration files and resolved overrides.
- Dataset manifest hash and split policy.
- Checkpoint hash.
- Environment versions and GPU model.
- Per-image metrics and aggregate values.
- Runtime and memory report.
- Ablation table.
- Failure-case gallery and notes.

Reports are generated under `outputs/`, which is excluded from Git. Only
curated, non-sensitive summaries should be committed, and never fabricated
values.
