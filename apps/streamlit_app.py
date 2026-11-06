"""SmartISP demonstration application.

The app is a functional shell around the public interfaces. It never displays
invented metrics: when a stage or model is still an interface placeholder, the
app reports that the implementation is pending.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

from smartisp import CFAPattern, ColorSpace, RawFrame, RGBImage
from smartisp.classical import build_default_pipeline
from smartisp.evaluation import MetricName
from smartisp.pipeline import HybridISPPipeline, HybridMode, HybridPipelineConfig


class _UnavailableRestorer:
    def restore(self, image: Any) -> Any:
        raise NotImplementedError(
            "the learned restoration model is not implemented in this scaffold"
        )


def _module_status() -> list[dict[str, str]]:
    return [
        {"area": "Classical ISP", "status": "interfaces only", "next": "stage 1"},
        {"area": "U-Net restoration", "status": "interfaces only", "next": "stage 2"},
        {"area": "Dataset provenance", "status": "discovery + guards", "next": "stage 2"},
        {"area": "Training loop", "status": "CLI dry-run only", "next": "stage 3"},
        {"area": "Metrics", "status": "interfaces only", "next": "stage 4"},
        {"area": "Evaluation analysis", "status": "planned matrix", "next": "stage 4"},
        {"area": "Colab workflow", "status": "documented + smoke test", "next": "stage 3"},
    ]


def _run_pipeline(
    image: RawFrame | RGBImage | None,
    mode: HybridMode,
    checkpoint: str,
) -> None:
    if image is None:
        st.warning("Upload an image first.")
        return
    if checkpoint:
        st.info(f"Checkpoint requested: {checkpoint}. Loading is not implemented yet.")
    classical = build_default_pipeline()
    restorer = _UnavailableRestorer()
    pipeline = HybridISPPipeline(
        classical=classical,
        restorer=restorer,
        config=HybridPipelineConfig(mode=mode),
    )
    try:
        result = pipeline.run(image)
    except NotImplementedError as exc:
        st.error(f"Pipeline stopped at an interface placeholder: {exc}")
        return
    st.success("Pipeline completed.")
    st.write(result)


def main() -> None:
    st.set_page_config(page_title="SmartISP", page_icon=None, layout="wide")
    st.title("SmartISP")
    st.caption("Intelligent hybrid smartphone image signal processing")

    st.warning(
        "Research scaffold: algorithm, training, and metric implementations are "
        "not complete. This app does not fabricate results."
    )

    with st.sidebar:
        st.header("Configuration")
        mode_label = st.selectbox(
            "Hybrid mode",
            options=[mode.value for mode in HybridMode],
            index=3,
        )
        mode = HybridMode(mode_label)
        checkpoint = st.text_input("Checkpoint path", value="")
        metrics = st.multiselect(
            "Metrics to report later",
            options=[metric.value for metric in MetricName],
            default=[metric.value for metric in MetricName],
        )
        st.divider()
        st.subheader("Domain policy")
        st.caption(
            "SIDD sRGB PNGs are display-referred images. They are not Bayer RAW. "
            "Use an explicit RAW dataset and CFA metadata for RAW processing."
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Project status")
        st.dataframe(_module_status(), hide_index=True, use_container_width=True)
    with right:
        st.subheader("Planned evaluation")
        st.write(
            "PSNR, SSIM, LPIPS, and CIEDE2000; runtime and memory percentiles; "
            "ablation variants and failure-case review."
        )
        st.caption(f"Selected metrics: {', '.join(metrics)}")

    st.divider()
    raw_tab, srgb_tab = st.tabs(["Bayer RAW input", "sRGB input"])

    with raw_tab:
        st.caption("Upload a single-channel .npy mosaic. Provide its true CFA metadata.")
        raw_file = st.file_uploader("Bayer .npy file", type=["npy"], key="raw")
        cfa_label = st.selectbox("CFA pattern", options=[pattern.value for pattern in CFAPattern])
        bit_depth = st.number_input("Bit depth", min_value=1, max_value=32, value=16)
        black_level = st.number_input("Black level", min_value=0.0, value=0.0)
        white_level = st.number_input("White level (0 means unspecified)", min_value=0.0, value=0.0)
        raw_image: RawFrame | None = None
        if raw_file is not None:
            mosaic = np.load(io.BytesIO(raw_file.read()), allow_pickle=False)
            if mosaic.ndim != 2:
                st.error(f"Expected a 2-D mosaic, got shape {mosaic.shape}.")
            else:
                st.write(f"Shape: {mosaic.shape}, dtype: {mosaic.dtype}")
                st.write(f"Min/Max: {float(mosaic.min())} / {float(mosaic.max())}")
                raw_image = RawFrame(
                    data=mosaic,
                    cfa=CFAPattern(cfa_label),
                    bit_depth=int(bit_depth),
                    black_level=float(black_level),
                    white_level=float(white_level) if white_level > 0 else None,
                    metadata={"source": raw_file.name},
                )
        if st.button("Run pipeline on RAW", key="run_raw"):
            _run_pipeline(raw_image, mode, checkpoint)

    with srgb_tab:
        st.caption(
            "Display-referred sRGB uploads are preview-only in this scaffold. "
            "They are never relabeled as Bayer RAW."
        )
        srgb_file = st.file_uploader("sRGB image", type=["png", "jpg", "jpeg", "webp"], key="srgb")
        if srgb_file is not None:
            image = Image.open(io.BytesIO(srgb_file.read())).convert("RGB")
            st.image(image, caption=srgb_file.name, use_container_width=True)
            array = np.asarray(image)
            preview = RGBImage(data=array, color_space=ColorSpace.SRGB)
            st.write(f"Shape: {preview.shape}")
            st.info(
                "The current classical pipeline starts from Bayer RAW. "
                "sRGB restoration requires a post-ISP model checkpoint and is not "
                "available yet."
            )

    st.divider()
    st.subheader("Remote execution")
    st.code(
        "colab run --gpu T4 scripts/check_remote_gpu.py\nbash scripts/colab_run.sh smartisp-t4 T4",
        language="bash",
    )


if __name__ == "__main__":
    main()
