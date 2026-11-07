# AGENTS.md

Repository-specific instructions for contributors and coding agents.

## Project status

SmartISP is at the **MVP demonstration stage**. Classical stages, the U-Net,
losses, training/validation, evaluation metrics, and Streamlit outputs are
implemented and covered by tests. The Colab T4 path is verified for a short
GPU training run (loss decreased 0.738 -> 0.386 over 3 epochs on 2026-11-07)
and for GPU inference from a saved checkpoint.

Honesty constraints remain binding: the local demo checkpoint is trained only
on deterministic synthetic pairs (`smartisp.data.synthetic`), never on real
sensor data. No SIDD benchmark numbers exist or may be invented. Anything run
without a checkpoint is explicitly labeled untrained.

## Environment

- Repository root: `/home/thushar/projects/SmartISP` inside WSL2.
- Windows access: `\\wsl.localhost\Ubuntu\home\thushar\projects\SmartISP`.
- Shell on Windows: PowerShell 5.1. Wrap Linux commands with
  `wsl -d Ubuntu -- bash -lc "cd /home/thushar/projects/SmartISP && <command>"`.
- Project Python: 3.12 (`.python-version`); `requires-python = ">=3.12,<3.14"`.
- Host Python observed: 3.14.4. Do not use the host interpreter for project
  installs; use `uv`, which creates the project virtual environment.
- `uv` observed: 0.13.7.
- Git remote: `https://github.com/thushar-chandra/SmartISP.git` (branch `main`).

## Dependency and setup commands

`pyproject.toml` is the authoritative dependency specification. Direct
dependencies are pinned to exact versions. `requirements/*.txt` are pinned views
used by Colab. `uv.lock` is committed; regenerate and commit it with dependency
changes.

A one-stop demo workflow (full extras assumed: `uv sync --extra dev --extra
app --extra torch`):

```bash
uv run python scripts/train_local_demo.py      # trains demo checkpoint on synthetic pairs
uv run streamlit run apps/streamlit_app.py     # demonstration UI
uv run python scripts/evaluate_demo.py         # classical vs learned vs hybrid metrics
```

Remote Colab runs (session management unchanged; use `requirements/colab_flat.txt`
because the CLI cannot resolve `-r base.txt` includes, it uploads a single
file):

```bash
colab exec -s smartisp-t4 --env SMARTISP_SYNTHETIC=true --env SMARTISP_EPOCHS=3 \
    -f scripts/colab_train.py
colab exec -s smartisp-t4 --env SMARTISP_CHECKPOINT=/content/checkpoints/smartisp_best.pt \
    --env SMARTISP_IMAGE=/content/data/val/input/srgb_scene_000.png \
    -f scripts/colab_inference.py
```

## Checks

Run from the repository root inside WSL2:

```bash
uv run pytest
uv run pytest tests/test_types.py::test_cfa_color_lookup -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv lock --check
uv build
```

A lightweight fallback that does not install the full project environment:

```bash
wsl -d Ubuntu -- bash -lc "cd /home/thushar/projects/SmartISP && PYTHONPATH=src uv run --no-project --with pytest --with pyyaml --with numpy pytest -q"
```

The fallback skips the PyTorch-dependent tests in `tests/test_models.py` when
PyTorch is absent.

## Package layout

- `src/smartisp/types.py`: `RawFrame`, `RGBImage`, `CFAPattern`, `ColorSpace`.
- `src/smartisp/classical/`: classical ISP stage contracts and composition.
- `src/smartisp/data/`: domain guards, pair discovery, preprocessing contracts.
- `src/smartisp/models/`: U-Net and loss contracts; torch imports are lazy at
  package level.
- `src/smartisp/pipeline/`: hybrid pipeline modes.
- `src/smartisp/training/`: CLI scaffolds with working `--dry-run`.
- `src/smartisp/evaluation/`: metric, efficiency, ablation, and failure
  contracts.
- `apps/streamlit_app.py`: demonstration shell; requires the `app` extra.
- `scripts/`: GPU checks and Colab helpers.
- `configs/`: default, Colab T4, and dataset manifest examples.

## Dataset rules

- SIDD sRGB PNGs are display-referred images, **not Bayer RAW**. Keep
  `input_domain: srgb`.
- RAW data requires an explicit CFA pattern and black/white levels.
- `smartisp.data.domains.validate_domain_for_suffix` is the enforcement point.
- Never commit `data/`, `datasets/`, checkpoints, outputs, `.env`, or
  `.streamlit/secrets.toml`.
- Split by scene to avoid leakage. Record the dataset manifest hash.

## Colab CLI workflow

The Colab CLI is installed in WSL2 at `/home/thushar/.local/bin/colab`; observed
version 0.7.5. Windows is not supported by the CLI. The following command shapes
were verified with `colab --help`, `colab new --help`, `colab install --help`,
`colab exec --help`, and `colab sessions`:

```bash
colab new -s smartisp-t4 --gpu T4
colab install -s smartisp-t4 -r requirements/colab.txt
colab exec -s smartisp-t4 --env SMARTISP_REPO_URL=https://github.com/thushar-chandra/SmartISP.git -f scripts/colab_smoke_test.py
colab log -s smartisp-t4 -o outputs/colab-smoke.md
colab stop -s smartisp-t4
```

Verified remote run (2026-11-06):

```text
colab run --gpu T4 scripts/check_remote_gpu.py
Python: 3.13.16
PyTorch: 2.14.1+cu130
CUDA available: True
GPU: Tesla T4
GPU memory: 14.56 GB
GPU computation: SUCCESS
session terminated automatically
```

Notes:

- `colab install -r` reads the local requirements file; `colab exec -f` sends
  the local script to the remote kernel. The remote VM does not automatically
  see the local repository, so the smoke test clones the pushed GitHub branch
  into `/content/SmartISP`. Commit and push scaffold changes before running it,
  or upload a tarball.
- The helper `bash scripts/colab_run.sh smartisp-t4 T4` wraps the sequence and
  stops the session on exit. It consumes GPU quota; run it deliberately.
- `colab run --gpu T4 scripts/check_remote_gpu.py` is the one-shot ephemeral
  check; it does not upload the repository or datasets.
- GPU allocation consumes compute units. Do not run it as an incidental test.
- Interactive commands (`colab auth`, `colab drivemount`, unpiped `colab repl`,
  `colab console`) require a TTY and must not be run from an agent.
- Always run `colab stop -s <session>` when finished.

## Implementation conventions

- Keep domain labels explicit; never infer RAW from shape or extension alone.
- Keep stage order explicit in `ClassicalISPPipeline`.
- Add tests for every interface change.
- Prefer dataclasses for configuration and typed protocols for stage contracts.
- Do not import PyTorch at package import time; keep `smartisp`, `smartisp.data`,
  `smartisp.evaluation`, and `smartisp.training.config` lightweight.
- Do not add code comments unless they are needed to explain non-obvious
  constraints; use docstrings for public behavior.
- Update this file when commands, package boundaries, or the Colab workflow
  change.

## Implemented status

Implemented and covered by tests: classical stages (Bayer preprocessor, white
balance, demosaic, color correction, denoise, tone mapping, sharpen, both a
full sensor pipeline and a conservative sRGB display pipeline), U-Net forward/
backward with automatic size padding, composite losses, checkpoint
save/load/resume, `PairedImageDataset` decoding/normalization/augmentation,
PSNR/SSIM/CIEDE2000/LPIPS, latency and memory measurement, the hybrid pipeline,
the Streamlit app, and both Colab scripts.

Not covered (by design, honesty constraints):

- No SIDD or other real-sensor training; the demo checkpoint is synthetic-only.
- RAW-domain (`PRE_ISP`) learned restoration has no RAW-trained checkpoint;
  `RGBRestorer` rejects `RawFrame` input on purpose.
- LPIPS transparently raises `ImportError` when the torch extra is absent.

The training and validation CLIs still support `--dry-run` and return exit
code `2` when inputs (data, checkpoint) are missing.
