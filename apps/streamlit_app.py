"""SmartISP demonstration application.

The app labels learned outputs as trained or untrained. It never invents
metrics: values are computed only when the user supplies a reference image.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any

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
    """List local checkpoint files, newest first."""
    root = Path(__file__).resolve().parents[1]
    return sorted(
        (root / "checkpoints").glob("*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


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
        output = pipeline.run(image)
    else:
        classical = None if mode is HybridMode.LEARNED_ONLY else build_display_pipeline()
        pipeline = HybridISPPipeline(
            classical=classical,
            restorer=restorer,
            config=HybridPipelineConfig(mode=mode),
        )
        output = pipeline.run(image)

    return {
        "image": output,
        "mode": mode.value,
        "trained": None if restorer is None else restorer.is_trained,
        "checkpoint": None if restorer is None else restorer.checkpoint,
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
    left, right = st.columns(2)
    with left:
        st.subheader("Before")
        st.image(_as_display_array(original), use_container_width=True)
    with right:
        st.subheader("After")
        st.image(_as_display_array(output), use_container_width=True)

    trained = result["trained"]
    if trained is True:
        provenance = _checkpoint_trained_on(result["checkpoint"])
        provenance_text = (
            "trained on SYNTHETIC pairs (not real sensor data)"
            if provenance == "synthetic_pairs"
            else f"trained checkpoint: {result['checkpoint']}"
        )
        st.success(f"Mode: {result['mode']} | {provenance_text}")
    elif trained is False:
        st.warning(
            "Mode: "
            f"{result['mode']} | UNTRAINED MODEL (random weights). "
            "This output is not a trained restoration result."
        )
    else:
        st.info(f"Mode: {result['mode']} | classical processing only")

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
            value=str(known[0]) if len(known) == 1 else "",
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
