# Remote GPU Workflow with the Google Colab CLI

## Platform

The official Colab CLI (`google-colab-cli`) supports Linux and macOS. Windows is
not supported. Use it inside WSL2:

```bash
wsl -d Ubuntu -- bash -lc "colab version"
```

Verified in this workspace:

```text
colab: 0.7.5
path: /home/thushar/.local/bin/colab
WSL: Ubuntu on WSL2
```

If the command is missing, install it with `uv` (recommended) or `pip`:

```bash
uv tool install google-colab-cli
```

## Authentication

The CLI defaults to interactive OAuth (`--auth oauth2`). Run an interactive
command once in a real terminal so the browser flow can complete:

```bash
colab sessions
```

Use `--auth adc` only if Application Default Credentials are already configured
for the required scopes. Never commit OAuth client files, tokens, or ADC
credentials. `colab auth` is a separate command for authenticating GCP services
inside a VM, not for storing credentials in the repository.

Interactive commands (`colab repl`, `colab console`, `colab auth`,
`colab drivemount`) expect a TTY. Do not run them from automated agents.

## Verified remote environment

On 2026-11-06 the one-shot GPU check completed successfully:

```text
command: colab run --gpu T4 scripts/check_remote_gpu.py
Python: 3.13.16
PyTorch: 2.14.1+cu130
CUDA available: True
GPU: Tesla T4
GPU memory: 14.56 GB
GPU computation: SUCCESS
session: run-1ba437, terminated automatically
```

The project supports Python 3.12 and 3.13; the remote runtime is 3.13.16.

## Supported accelerators

`colab new --gpu` accepts `T4`, `L4`, `G4`, `H100`, or `A100`. The project target
is `T4`. Accelerator availability is tier-gated; a quota or entitlement error
means the account cannot allocate that accelerator at that time. Do not silently
fall back to a different GPU in an experiment report.

## Standard session workflow

From WSL2, in the repository root:

```bash
colab new -s smartisp-t4 --gpu T4
colab install -s smartisp-t4 -r requirements/colab.txt
colab exec -s smartisp-t4 --env SMARTISP_REPO_URL=https://github.com/thushar-chandra/SmartISP.git -f scripts/colab_smoke_test.py
colab log -s smartisp-t4 -o outputs/colab-smoke.md
colab stop -s smartisp-t4
```

What each step does:

- `colab new` provisions a fresh T4 runtime.
- `colab install -r` installs the pinned packages on the remote VM using `uv`
  with a `pip` fallback.
- `colab exec -f` sends the local Python file to the remote kernel and executes
  it. The remote process does not automatically see the local repository, so the
  smoke test clones the repository into `/content/SmartISP` and imports it from
  `src/`.
- `colab log -o` exports the session history for the experiment record.
- `colab stop` releases the VM and its compute units.

The helper script wraps the sequence and stops the session on exit:

```bash
bash scripts/colab_run.sh smartisp-t4 T4
```

This helper consumes GPU quota. Run it deliberately.

## Ephemeral one-shot job

For a short GPU check with automatic teardown:

```bash
colab run --gpu T4 scripts/check_remote_gpu.py
```

`colab run` provisions a fresh VM, runs one local Python script, forwards
arguments, and tears the VM down unless `--keep` is supplied. It does not clone
the repository or upload datasets. Use it for self-contained checks.

## The smoke test

`scripts/colab_smoke_test.py`:

1. Verifies Python, PyTorch, CUDA, and the T4 device.
2. Clones the repository if `smartisp` is not importable.
3. Imports the package and prints its version.
4. Instantiates the U-Net and runs a small CUDA forward pass on it.
5. Runs a small CUDA matrix multiplication.
6. Prints `SMOKE TEST PASSED` only when all checks pass.

It does not train and does not report accuracy.

The clone uses the repository's pushed GitHub branch. Commit and push scaffold
changes before running the smoke test, or upload the repository as a tarball and
adjust the import path.

## Data transfer

Datasets are not committed to Git. Typical options:

1. Upload a single file:
   ```bash
   colab upload -s smartisp-t4 data/sample.npy /content/data/sample.npy
   ```
2. Upload a directory as a tarball, then extract remotely:
   ```bash
   tar -czf /tmp/session-data.tar.gz -C data sidd-srgb
   colab upload -s smartisp-t4 /tmp/session-data.tar.gz /content/session-data.tar.gz
   ```
   Extract it with a Python or shell step executed through the session.
3. Mount Google Drive (interactive; requires a TTY):
   ```bash
   colab drivemount -s smartisp-t4
   ```
   The default mount point is `/content/drive`.

Keep large datasets on remote or mounted storage. Do not commit them.

## Checkpoints and artifacts

Download checkpoints before stopping the session:

```bash
colab download -s smartisp-t4 /content/checkpoints/model.pth ./checkpoints/model.pth
colab download -s smartisp-t4 /content/outputs/report.json ./outputs/report.json
```

Checkpoints and outputs are ignored by Git. Record the checkpoint hash and the
session log alongside any reported result.

## Remote training workflow (planned)

Training is not implemented yet. When stage 3 lands, the intended workflow is:

```bash
colab new -s smartisp-t4 --gpu T4
colab install -s smartisp-t4 -r requirements/colab.txt
colab exec -s smartisp-t4 -f scripts/colab_bootstrap_train.py
colab status -s smartisp-t4
colab log -s smartisp-t4 -o outputs/colab-train.md
colab download -s smartisp-t4 /content/checkpoints/final.pth ./checkpoints/final.pth
colab stop -s smartisp-t4
```

The bootstrap script does not exist yet. Do not treat this planned sequence as a
working training pipeline.

## Session management

```bash
colab sessions
colab status -s smartisp-t4
colab usage
colab restart-kernel -s smartisp-t4
colab stop -s smartisp-t4
```

Always stop sessions when finished. A running session can continue to consume
compute units even when no command is executing.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `colab: command not found` | Add `~/.local/bin` to `PATH` or run inside WSL login shell |
| `400` on `colab new --gpu T4` | Account quota or entitlement; do not silently change GPU |
| `smartisp` import fails | Repository clone failed or `src/` was not added to `sys.path` |
| CUDA unavailable | Verify `colab status` and rerun the smoke test |
| Timeout during `colab exec` | Increase `--timeout`; default is 30 seconds |
| Dataset missing | Upload, extract, or mount it before running training |

## Security rules

- Never commit tokens, OAuth client files, service-account keys, or `.env`.
- Never embed signed dataset URLs in tracked files.
- Keep model checkpoints out of Git.
- Export only non-sensitive session logs and curated reports.
