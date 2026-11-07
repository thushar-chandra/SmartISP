"""SmartISP demonstration application.

The app labels learned outputs as trained or untrained. It never invents
metrics: values are computed only when the user supplies a reference image.
"""

from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from smartisp import CFAPattern, ColorSpace, RawFrame, RGBImage
from smartisp.classical import build_default_pipeline, build_display_pipeline
from smartisp.evaluation import MetricConfig, MetricName, evaluate_pair
from smartisp.pipeline import HybridISPPipeline, HybridMode, HybridPipelineConfig

_DEFAULT_METRICS = (MetricName.PSNR, MetricName.SSIM, MetricName.DELTA_E_2000)


def _module_status() -> list[dict[str, str]]:
    return [
        {"area": "Classical ISP", "status": "implemented", "next": "tune on real RAW"},
        {"area": "U-Net restoration", "status": "implemented", "next": "train checkpoint"},
        {"area": "Dataset decoding", "status": "implemented", "next": "validate on real pairs"},
        {"area": "Training loop", "status": "implemented", "next": "Colab T4 run"},
        {"area": "Metrics", "status": "PSNR, SSIM, CIEDE2000, LPIPS", "next": "report on test set"},
        {
            "area": "Evaluation analysis",
            "status": "ablation + failure helpers",
            "next": "dataset run",
        },
        {"area": "Colab workflow", "status": "train + inference scripts", "next": "run remotely"},
    ]


def _known_checkpoints() -> list[Path]:
    """List local checkpoint files, prioritizing best checkpoints, then newest."""
    root = Path(__file__).resolve().parents[1]
    candidates = list((root / "checkpoints").glob("*.pt"))

    def sort_key(path: Path) -> tuple[int, float]:
        is_best = 0 if "best" in path.stem.lower() else 1
        return (is_best, -path.stat().st_mtime)

    return sorted(candidates, key=sort_key)


def _checkpoint_trained_on(checkpoint: Path | None) -> str:
    """Best-effort training-provenance hint from names and side-car files."""
    if checkpoint is None:
        return "unknown"
    name = checkpoint.stem.lower()
    if "synthetic" in name:
        return "synthetic_pairs"
    if any(token in name for token in {"sidd", "mir"}):
        return "real_sensor_pairs"
    for candidate in (checkpoint.parent, Path.cwd() / "outputs"):
        sidecar = candidate / f"{checkpoint.stem}_training_summary.json"
        if sidecar.is_file():
            try:
                import json

                data = json.loads(sidecar.read_text(encoding="utf-8"))
                return str(data.get("trained_on", sidecar.name))
            except Exception:
                return sidecar.name
    return "unknown"


def _as_display_array(image: RGBImage | RawFrame) -> np.ndarray:
    data = np.asarray(image.data)
    if isinstance(image, RawFrame):
        preview = data.astype(np.float32)
        minimum = float(preview.min()) if preview.size else 0.0
        maximum = float(preview.max()) if preview.size else 1.0
        scaled = (preview - minimum) / max(maximum - minimum, 1e-8)
        return np.repeat((scaled * 255).astype(np.uint8)[..., None], 3, axis=-1)
    display = data.astype(np.float32)
    if display.max(initial=0.0) <= 1.5:
        display = display * 255.0
    return np.clip(display, 0.0, 255.0).astype(np.uint8)


def _png_bytes(image: RGBImage | RawFrame) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(_as_display_array(image)).save(buffer, format="PNG")
    return buffer.getvalue()


def _load_uploaded_rgb(uploaded: Any) -> RGBImage:
    rgb = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    return RGBImage(
        data=np.asarray(rgb).astype(np.float32) / 255.0,
        color_space=ColorSpace.SRGB,
        bit_depth=8,
        metadata={"source": uploaded.name},
    )


def _load_uploaded_raw(
    uploaded: Any,
    cfa: CFAPattern,
    bit_depth: int,
    black_level: float,
    white_level: float,
) -> RawFrame:
    mosaic = np.load(io.BytesIO(uploaded.read()), allow_pickle=False)
    if mosaic.ndim != 2:
        raise ValueError(f"expected a 2-D Bayer mosaic, got shape {mosaic.shape}")
    return RawFrame(
        data=mosaic,
        cfa=cfa,
        bit_depth=int(bit_depth),
        black_level=float(black_level),
        white_level=float(white_level) if white_level > 0 else None,
        metadata={"source": uploaded.name},
    )


def _checkpoint_path(path_text: str, uploaded: Any) -> Path | None:
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".pt"
        descriptor, name = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(uploaded.read())
        return Path(name)
    if path_text.strip():
        return Path(path_text.strip()).expanduser()
    return None


def _build_restorer(checkpoint: Path | None, device: str, allow_untrained: bool) -> Any:
    try:
        from smartisp.models import build_restorer
    except ImportError as exc:
        raise RuntimeError("learned restoration requires PyTorch and the 'torch' extra") from exc
    if checkpoint is None and not allow_untrained:
        raise RuntimeError(
            "no checkpoint provided; enable 'Use untrained model' to run random weights"
        )
    if checkpoint is not None and not checkpoint.is_file():
        raise RuntimeError(f"checkpoint does not exist: {checkpoint}")
    return build_restorer(checkpoint, device=device)


def _generate_difference_map(
    before: np.ndarray,
    after: np.ndarray,
    amplification: float = 5.0,
    colormap: str = "inferno",
) -> np.ndarray:
    diff = np.abs(after.astype(np.float32) - before.astype(np.float32))
    diff_mag = np.mean(diff, axis=-1) * amplification
    clipped = np.clip(diff_mag, 0.0, 255.0).astype(np.uint8)
    if colormap == "grayscale":
        return np.repeat(clipped[..., None], 3, axis=-1)
    cm = cv2.COLORMAP_INFERNO if colormap == "inferno" else cv2.COLORMAP_JET
    colored = cv2.applyColorMap(clipped, cm)
    return colored[..., ::-1]


def _run_pipeline(
    image: RawFrame | RGBImage | None,
    mode: HybridMode,
    checkpoint: Path | None = None,
    *,
    device: str = "auto",
    allow_untrained: bool = False,
) -> dict[str, Any]:
    if image is None:
        raise ValueError("upload an image first")

    restorer = None
    if mode is not HybridMode.CLASSICAL_ONLY:
        restorer = _build_restorer(checkpoint, device, allow_untrained)

    t0 = time.perf_counter()
    if isinstance(image, RawFrame):
        if mode in {HybridMode.LEARNED_ONLY, HybridMode.PRE_ISP}:
            raise ValueError(
                "this restorer accepts RGB images; RAW-domain modes need a RAW "
                "checkpoint and adapter"
            )
        pipeline = HybridISPPipeline(
            classical=build_default_pipeline(),
            restorer=restorer,
            config=HybridPipelineConfig(mode=mode),
        )
        output, stages = pipeline.run_with_trace(image)
    else:
        classical = None if mode is HybridMode.LEARNED_ONLY else build_display_pipeline()
        pipeline = HybridISPPipeline(
            classical=classical,
            restorer=restorer,
            config=HybridPipelineConfig(mode=mode),
        )
        output, stages = pipeline.run_with_trace(image)
    total_time_ms = (time.perf_counter() - t0) * 1000.0

    in_arr = _as_display_array(image)
    out_arr = _as_display_array(output)
    diff = np.abs(out_arr.astype(np.float32) - in_arr.astype(np.float32))
    diff_stats = {
        "mean_diff_255": float(np.mean(diff)),
        "max_diff_255": float(np.max(diff)),
        "pct_changed": float(np.mean(diff >= 1.0) * 100.0),
        "input_min": float(np.min(in_arr)),
        "input_max": float(np.max(in_arr)),
        "output_min": float(np.min(out_arr)),
        "output_max": float(np.max(out_arr)),
    }

    return {
        "image": output,
        "mode": mode.value,
        "trained": None if restorer is None else restorer.is_trained,
        "checkpoint": None if restorer is None else restorer.checkpoint,
        "stages": stages,
        "latency_ms": total_time_ms,
        "device": device if restorer is None else str(restorer.device),
        "diff_stats": diff_stats,
    }


def _metrics_for(
    result: RGBImage, reference: RGBImage, names: list[MetricName]
) -> dict[str, float]:
    prediction = _as_display_array(result).astype(np.float32) / 255.0
    target = _as_display_array(reference).astype(np.float32) / 255.0
    if prediction.shape != target.shape:
        raise ValueError(
            f"reference shape {target.shape} does not match output shape {prediction.shape}"
        )
    return evaluate_pair(prediction, target, names, MetricConfig())


def _render_result(
    result: dict[str, Any],
    original: RawFrame | RGBImage,
    reference: RGBImage | None,
    metric_names: list[MetricName],
) -> None:
    output: RGBImage = result["image"]
    before_arr = _as_display_array(original)
    after_arr = _as_display_array(output)

    left, right = st.columns(2)
    with left:
        st.subheader("Before")
        st.image(before_arr, use_container_width=True)
    with right:
        st.subheader("After")
        st.image(after_arr, use_container_width=True)

    trained = result.get("trained")
    checkpoint = result.get("checkpoint")
    provenance = _checkpoint_trained_on(Path(checkpoint) if checkpoint else None)

    if trained is True:
        if provenance == "synthetic_pairs":
            st.info(
                f"**Mode:** `{result['mode']}` | **Training Provenance:** "
                "Trained on **DETERMINISTIC SYNTHETIC PAIRS** (`smartisp.data.synthetic`), "
                "not real smartphone sensor data (e.g. SIDD). In hybrid mode, high-frequency "
                "sensor noise is suppressed by the classical bilateral denoiser while the learned "
                "model restores contrast and structural details."
            )
        else:
            st.success(
                f"**Mode:** `{result['mode']}` | **Training Provenance:** "
                f"Trained checkpoint (`{checkpoint}`)"
            )
    elif trained is False:
        st.warning(
            f"**Mode:** `{result['mode']}` | **UNTRAINED MODEL** (random weights). "
            "This output is not a trained restoration result."
        )
    else:
        st.info(
            f"**Mode:** `{result['mode']}` | "
            "Classical ISP processing only (no neural weights used)."
        )

    # Diagnostics section
    with st.expander("Diagnostics & Image Restoration Quality", expanded=True):
        diff_stats = result.get("diff_stats", {})
        latency = result.get("latency_ms", 0.0)
        stages = result.get("stages", [])
        device_used = result.get("device", "cpu")

        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        with d_col1:
            st.metric("Processing Time", f"{latency:.1f} ms")
            st.caption(f"Device: `{device_used}`")
        with d_col2:
            st.metric("Mean Pixel Diff", f"{diff_stats.get('mean_diff_255', 0.0):.2f} / 255")
            st.caption(f"Max Diff: {diff_stats.get('max_diff_255', 0.0):.0f} / 255")
        with d_col3:
            st.metric("Pixels Changed", f"{diff_stats.get('pct_changed', 0.0):.1f}%")
            st.caption("Threshold: >= 1/255 level")
        with d_col4:
            in_min = diff_stats.get("input_min", 0.0)
            in_max = diff_stats.get("input_max", 255.0)
            out_min = diff_stats.get("output_min", 0.0)
            out_max = diff_stats.get("output_max", 255.0)
            st.metric("Dynamic Range", f"[{out_min:.0f}, {out_max:.0f}]")
            st.caption(f"Input: [{in_min:.0f}, {in_max:.0f}]")

        st.markdown("**Executed Stages:** " + " -> ".join(f"`{s}`" for s in stages))
        if checkpoint:
            status_text = "Trained" if trained else "Untrained"
            st.caption(f"Checkpoint: `{checkpoint}` | Status: {status_text}")

        st.divider()
        st.subheader("Restoration Difference Visualization")
        st.caption(
            "Visualizes absolute pixel differences (|After - Before|). "
            "On high-resolution photos, subtle pixel-level restoration (2-8 levels) is easily lost "
            "when downsampled in web previews. Amplification highlights where restoration occurred."
        )
        ctrl_col1, ctrl_col2 = st.columns(2)
        with ctrl_col1:
            amplify = st.slider(
                "Difference Amplification",
                min_value=1.0,
                max_value=20.0,
                value=5.0,
                step=1.0,
            )
        with ctrl_col2:
            cmap = st.selectbox("Colormap", ["inferno", "jet", "grayscale"])

        diff_map = _generate_difference_map(
            before_arr, after_arr, amplification=amplify, colormap=cmap
        )
        st.image(
            diff_map,
            caption=f"Absolute Difference Map ({amplify:.0f}x Amplification, {cmap})",
            use_container_width=True,
        )

        # 100% Zoom Crop (1:1 Native Pixel Inspection)
        st.subheader("100% Zoom Crop (1:1 Pixel Inspection)")
        st.caption(
            "Inspects a native crop without browser downsampling to verify fine-grain "
            "noise removal and edge crispness."
        )
        h, w = before_arr.shape[:2]
        crop_size = min(256, h, w)
        crop_positions = {
            "Center": (max(0, (h - crop_size) // 2), max(0, (w - crop_size) // 2)),
            "Top-Left": (0, 0),
            "Top-Right": (0, max(0, w - crop_size)),
            "Bottom-Left": (max(0, h - crop_size), 0),
            "Bottom-Right": (max(0, h - crop_size), max(0, w - crop_size)),
        }
        pos = st.selectbox("Crop Region", list(crop_positions.keys()))
        y0, x0 = crop_positions[pos]
        crop_before = before_arr[y0 : y0 + crop_size, x0 : x0 + crop_size]
        crop_after = after_arr[y0 : y0 + crop_size, x0 : x0 + crop_size]
        crop_diff = diff_map[y0 : y0 + crop_size, x0 : x0 + crop_size]

        z_col1, z_col2, z_col3 = st.columns(3)
        with z_col1:
            st.caption("Before (1:1 Crop)")
            st.image(crop_before, use_container_width=True)
        with z_col2:
            st.caption("After (1:1 Crop)")
            st.image(crop_after, use_container_width=True)
        with z_col3:
            st.caption(f"Difference ({amplify:.0f}x)")
            st.image(crop_diff, use_container_width=True)

    if reference is not None:
        try:
            values = _metrics_for(output, reference, metric_names)
            st.subheader("Metrics against provided reference")
            columns = st.columns(len(values))
            for column, (name, value) in zip(columns, values.items(), strict=False):
                higher = name in {"psnr", "ssim"}
                delta = None
                column.metric(name.upper(), f"{value:.4f}", delta=delta)
                column.caption("higher is better" if higher else "lower is better")
        except ImportError as exc:
            st.error(f"Metric unavailable: {exc}")
        except ValueError as exc:
            st.error(str(exc))
    else:
        st.caption("Upload a reference image to compute PSNR, SSIM, and color difference.")

    st.download_button(
        "Download processed PNG",
        data=_png_bytes(output),
        file_name="smartisp_output.png",
        mime="image/png",
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title="SmartISP", page_icon=None, layout="wide")
    st.title("SmartISP")
    st.caption("Hybrid classical and learned smartphone image signal processing")

    with st.sidebar:
        st.header("Configuration")
        source = st.radio("Input type", ["sRGB image", "Bayer RAW (.npy)"])
        if source == "sRGB image":
            mode_options = [
                HybridMode.CLASSICAL_ONLY,
                HybridMode.LEARNED_ONLY,
                HybridMode.POST_ISP,
                HybridMode.PRE_ISP,
            ]
            default_mode = HybridMode.POST_ISP
        else:
            mode_options = [HybridMode.CLASSICAL_ONLY, HybridMode.POST_ISP]
            default_mode = HybridMode.POST_ISP
        mode = st.selectbox(
            "Processing mode",
            options=mode_options,
            index=mode_options.index(default_mode),
            format_func=lambda item: item.value,
        )

        checkpoint_upload = st.file_uploader(
            "Checkpoint (.pt/.pth)", type=["pt", "pth"], key="checkpoint_file"
        )
        known = _known_checkpoints()
        checkpoint_text = st.text_input(
            "Checkpoint path",
            value=str(known[0]) if known else "",
            placeholder=", ".join(str(path) for path in known) or "path/to/ckpt.pt",
        )
        if known:
            st.caption("Detected local checkpoints: " + ", ".join(f"`{path}`" for path in known))
        else:
            st.caption(
                "No local checkpoints found under `checkpoints/`. Train one with "
                "`scripts/train_local_demo.py` or `scripts/colab_train.py`, or "
                "upload a .pt file."
            )
            st.warning("Without a checkpoint, learned modes use untrained random weights.")
        allow_untrained = st.checkbox(
            "Use untrained model (random weights) when no checkpoint is given",
            value=False,
        )
        device = st.selectbox("Device", ["auto", "cpu", "cuda"])
        metric_names = st.multiselect(
            "Metrics (when a reference is supplied)",
            options=[metric.value for metric in MetricName],
            default=[metric.value for metric in _DEFAULT_METRICS],
        )

        st.divider()
        st.subheader("Domain policy")
        st.caption(
            "SIDD sRGB PNGs are display-referred images, not Bayer RAW. RAW input "
            "requires an explicit CFA pattern and black/white levels."
        )

    with st.expander("Project status", expanded=False):
        st.dataframe(_module_status(), hide_index=True, use_container_width=True)

    original: RawFrame | RGBImage | None = None
    if source == "Bayer RAW (.npy)":
        raw_file = st.file_uploader("Bayer mosaic (.npy)", type=["npy"], key="raw")
        cfa_label = st.selectbox("CFA pattern", options=[pattern.value for pattern in CFAPattern])
        bit_depth = st.number_input("Bit depth", min_value=1, max_value=32, value=16)
        black_level = st.number_input("Black level", min_value=0.0, value=0.0)
        white_level = st.number_input("White level (0 means unspecified)", min_value=0.0, value=0.0)
        if raw_file is not None:
            try:
                original = _load_uploaded_raw(
                    raw_file,
                    CFAPattern(cfa_label),
                    int(bit_depth),
                    float(black_level),
                    float(white_level),
                )
            except (ValueError, OSError) as exc:
                st.error(str(exc))
            if original is not None:
                data = np.asarray(original.data)
                st.write(
                    f"Shape: {data.shape}, dtype: {data.dtype}, "
                    f"min/max: {float(data.min())}/{float(data.max())}"
                )
    else:
        srgb_file = st.file_uploader(
            "Image", type=["png", "jpg", "jpeg", "webp", "bmp"], key="srgb"
        )
        if srgb_file is not None:
            try:
                original = _load_uploaded_rgb(srgb_file)
            except OSError as exc:
                st.error(str(exc))
            if original is not None:
                st.image(
                    Image.fromarray(_as_display_array(original)),
                    caption=srgb_file.name,
                    use_container_width=True,
                )

    reference: RGBImage | None = None
    with st.expander("Reference image for metrics (optional)"):
        reference_file = st.file_uploader(
            "Reference", type=["png", "jpg", "jpeg", "webp", "bmp"], key="reference"
        )
        if reference_file is not None:
            try:
                reference = _load_uploaded_rgb(reference_file)
            except OSError as exc:
                st.error(str(exc))

    if st.button("Run SmartISP", type="primary", use_container_width=True):
        checkpoint = _checkpoint_path(checkpoint_text, checkpoint_upload)
        try:
            result = _run_pipeline(
                original,
                mode,
                checkpoint,
                device=device,
                allow_untrained=allow_untrained,
            )
        except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
            st.error(str(exc))
        else:
            selected = [MetricName(name) for name in metric_names]
            _render_result(result, original, reference, selected)
    elif original is not None:
        st.info("Choose a mode and press 'Run SmartISP'.")

    st.divider()
    st.subheader("Remote execution")
    st.code(
        "python scripts/colab_train.py --synthetic --epochs 1\n"
        "python scripts/colab_inference.py --checkpoint checkpoints/*_best.pt --image sample.png",
        language="bash",
    )


if __name__ == "__main__":
    main()
