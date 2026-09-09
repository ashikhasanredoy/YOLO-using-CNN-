"""
Adaptive Image Enhancement for Robust Object Detection — Interactive Streamlit Dashboard
Allows users to upload driving images or select benchmark samples across adverse weather conditions,
runs real-time CNN condition classification and YOLO detection across Baseline, Fixed, and Adaptive pipelines,
and displays side-by-side visual comparisons, detection diagnostics, and research analytics.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import yaml
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import streamlit as st
from ultralytics import YOLO

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.condition_cnn import build_condition_classifier, get_device
from src.data.dataset import get_transforms
from src.data.metadata import CONDITION_NAMES
from src.enhancement.fixed import FixedCLAHEEnhancer
from src.enhancement.adaptive import AdaptiveEnhancer
from src.evaluation.visualization import draw_bounding_boxes, COLOR_PALETTE


# Page configuration
st.set_page_config(
    page_title="Adaptive YOLO Research Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3b82f6, #10b981, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .badge-clear { background-color: #22c55e; color: white; padding: 3px 8px; border-radius: 6px; font-weight: 600; }
    .badge-rain { background-color: #3b82f6; color: white; padding: 3px 8px; border-radius: 6px; font-weight: 600; }
    .badge-fog { background-color: #a855f7; color: white; padding: 3px 8px; border-radius: 6px; font-weight: 600; }
    .badge-night { background-color: #1e293b; color: #f8fafc; border: 1px solid #64748b; padding: 3px 8px; border-radius: 6px; font-weight: 600; }
    .badge-dawn_dusk { background-color: #f59e0b; color: white; padding: 3px 8px; border-radius: 6px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Loads and caches PyTorch CNN and YOLO models."""
    device = torch.device("cpu")  # CPU safe for local Streamlit serving
    
    # YOLO
    yolo_model = YOLO("yolo11n.pt")

    # ResNet-18 Condition Classifier
    ckpt_path = PROJECT_ROOT / "models" / "condition_classifier_best.pt"
    classifier = build_condition_classifier("resnet18", num_classes=5, pretrained=False)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        classifier.load_state_dict(ckpt.get("model_state_dict", ckpt))
    classifier.to(device)
    classifier.eval()

    transform = get_transforms(image_size=224, is_train=False)

    return yolo_model, classifier, transform, device


yolo_model, condition_classifier, cnn_transform, compute_device = load_models()

# Load config
config_file = PROJECT_ROOT / "configs" / "config.yaml"
if config_file.exists():
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
else:
    config = {}


def predict_condition(img_bgr: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
    """Runs CNN forward pass and returns predicted class and all probabilities."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    tensor = cnn_transform(img_pil).unsqueeze(0).to(compute_device)

    with torch.no_grad():
        logits = condition_classifier(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_id = int(np.argmax(probs))
    pred_name = CONDITION_NAMES.get(pred_id, "clear")
    confidence = float(probs[pred_id])

    prob_dict = {CONDITION_NAMES[i]: float(probs[i]) for i in range(5)}
    return pred_name, confidence, prob_dict


def run_yolo_detection(img_bgr: np.ndarray, conf_thresh: float, iou_thresh: float) -> Tuple[List[Dict[str, Any]], float]:
    """Runs Ultralytics YOLO inference."""
    t0 = time.perf_counter()
    results = yolo_model.predict(
        source=img_bgr,
        conf=conf_thresh,
        iou=iou_thresh,
        verbose=False
    )[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    dets = []
    for box in results.boxes:
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        cls_name = yolo_model.names.get(cls_id, str(cls_id))
        dets.append({
            "bbox_xyxy": xyxy,
            "confidence": conf,
            "class_id": cls_id,
            "class_name": cls_name
        })
    return dets, latency_ms


# Header
st.markdown('<div class="main-header">Adaptive Image Enhancement for Object Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Empirical benchmarking of Baseline vs. Fixed CLAHE vs. Adaptive CNN-Routed Enhancement on adverse weather conditions (BDD100K).</div>', unsafe_allow_html=True)

# Tabs
tab_demo, tab_analytics, tab_about = st.tabs(["🧪 Interactive Pipeline Testing", "📊 Research Analytics & Plots", "📖 Methodology & Details"])

with st.sidebar:
    st.header("⚙️ Experiment Controls")
    
    st.subheader("YOLO Inference Settings")
    conf_thresh = st.slider("Confidence Threshold", 0.05, 0.90, 0.25, 0.05)
    iou_thresh = st.slider("IoU NMS Threshold", 0.10, 0.90, 0.50, 0.05)

    st.subheader("Enhancement Parameters")
    fixed_clip = st.slider("Fixed CLAHE Clip Limit", 0.5, 5.0, 2.0, 0.5)
    night_gamma = st.slider("Night Gamma (Low-Light)", 0.3, 1.0, 0.60, 0.05)
    rain_strength = st.slider("Rain Unsharp Strength", 0.5, 2.5, 1.2, 0.1)
    
    st.markdown("---")
    st.markdown("**Backbone Models:**")
    st.markdown("• Classifier: `ResNet-18` (5 Classes)")
    st.markdown("• Detector: `Ultralytics YOLO11n`")


with tab_demo:
    col_input, col_preset = st.columns([2, 1])

    with col_input:
        uploaded_file = st.file_uploader(
            "Upload an adverse weather or driving image (JPG / PNG):",
            type=["jpg", "jpeg", "png"]
        )

    with col_preset:
        # Pre-loaded benchmark samples from local dataset
        image_dir = PROJECT_ROOT / "data" / "bdd100k" / "images"
        preset_choice = None
        if image_dir.exists():
            sample_files = list(image_dir.glob("*.jpg"))[:10]
            if sample_files:
                sample_names = ["None (Upload custom)"] + [f.name for f in sample_files]
                preset_choice = st.selectbox("Or choose a benchmark test sample:", sample_names)

    img_bgr = None

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.success(f"Loaded custom uploaded image: {uploaded_file.name} ({img_bgr.shape[1]}x{img_bgr.shape[0]})")
    elif preset_choice and preset_choice != "None (Upload custom)":
        preset_path = image_dir / preset_choice
        img_bgr = cv2.imread(str(preset_path))
        st.info(f"Loaded benchmark sample: {preset_choice}")

    if img_bgr is not None:
        # 1. Condition Classification
        pred_condition, conf, probs = predict_condition(img_bgr)
        
        # Display condition classification card
        st.markdown("### 🌤️ Condition Classifier Prediction")
        c_col1, c_col2 = st.columns([1, 2])
        with c_col1:
            badge_class = f"badge-{pred_condition}"
            st.markdown(f"**Predicted Condition:** <span class='{badge_class}'>{pred_condition.upper()}</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** `{conf * 100:.2f}%`")
        with c_col2:
            df_probs = pd.DataFrame({
                "Condition": list(probs.keys()),
                "Probability": list(probs.values())
            })
            st.bar_chart(df_probs.set_index("Condition"), height=150)

        # 2. Run All 3 Pipelines
        st.markdown("### 🔬 Comparative Pipeline Detection")

        # Pipeline A: Baseline
        with st.spinner("Processing Pipeline A (Baseline)..."):
            dets_a, latency_a = run_yolo_detection(img_bgr, conf_thresh, iou_thresh)
            vis_a = draw_bounding_boxes(img_bgr, dets_a, title_text="Pipeline A: Baseline (No Enhancement)")

        # Pipeline B: Fixed CLAHE
        with st.spinner("Processing Pipeline B (Fixed CLAHE)..."):
            t_enh_b0 = time.perf_counter()
            fixed_enhancer = FixedCLAHEEnhancer(clip_limit=fixed_clip)
            enhanced_b = fixed_enhancer.enhance(img_bgr)
            t_enh_b = (time.perf_counter() - t_enh_b0) * 1000.0
            dets_b, latency_b = run_yolo_detection(enhanced_b, conf_thresh, iou_thresh)
            vis_b = draw_bounding_boxes(enhanced_b, dets_b, title_text=f"Pipeline B: Fixed CLAHE (Clip={fixed_clip})")

        # Pipeline C: Adaptive
        with st.spinner("Processing Pipeline C (Adaptive Router)..."):
            enh_cfg = config.get("enhancement", {})
            enh_cfg["night"]["gamma"] = night_gamma
            enh_cfg["rain"]["unsharp_strength"] = rain_strength
            adaptive_enhancer = AdaptiveEnhancer(enh_cfg)

            t_enh_c0 = time.perf_counter()
            targeted_enhancer = adaptive_enhancer.get_enhancer(pred_condition)
            enhanced_c = targeted_enhancer.enhance(img_bgr)
            t_enh_c = (time.perf_counter() - t_enh_c0) * 1000.0
            dets_c, latency_c = run_yolo_detection(enhanced_c, conf_thresh, iou_thresh)
            vis_c = draw_bounding_boxes(enhanced_c, dets_c, title_text=f"Pipeline C: Adaptive ({targeted_enhancer.name})")

        # 3-Column Visual Display
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.subheader("Pipeline A: Baseline")
            st.image(cv2.cvtColor(vis_a, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Detections: **{len(dets_a)}** | Inference: **{latency_a:.1f}ms**")

        with col_b:
            st.subheader("Pipeline B: Fixed CLAHE")
            st.image(cv2.cvtColor(vis_b, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Detections: **{len(dets_b)}** | Enh: **{t_enh_b:.1f}ms** | Inf: **{latency_b:.1f}ms**")

        with col_c:
            st.subheader(f"Pipeline C: Adaptive ({pred_condition})")
            st.image(cv2.cvtColor(vis_c, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Detections: **{len(dets_c)}** | Enh: **{t_enh_c:.1f}ms** | Inf: **{latency_c:.1f}ms**")

        # Detection Summary Table
        st.markdown("#### 📋 Detections Breakdown")
        df_summary = pd.DataFrame([
            {
                "Pipeline": "Pipeline A (Baseline)",
                "Enhancement": "None",
                "Objects Detected": len(dets_a),
                "Enhancement Time (ms)": 0.0,
                "YOLO Latency (ms)": round(latency_a, 1),
                "Total Latency (ms)": round(latency_a, 1)
            },
            {
                "Pipeline": "Pipeline B (Fixed CLAHE)",
                "Enhancement": f"Fixed CLAHE (clip={fixed_clip})",
                "Objects Detected": len(dets_b),
                "Enhancement Time (ms)": round(t_enh_b, 1),
                "YOLO Latency (ms)": round(latency_b, 1),
                "Total Latency (ms)": round(t_enh_b + latency_b, 1)
            },
            {
                "Pipeline": "Pipeline C (Adaptive)",
                "Enhancement": targeted_enhancer.name,
                "Objects Detected": len(dets_c),
                "Enhancement Time (ms)": round(t_enh_c, 1),
                "YOLO Latency (ms)": round(latency_c, 1),
                "Total Latency (ms)": round(t_enh_c + latency_c, 1)
            }
        ])
        st.dataframe(df_summary, use_container_width=True)
    else:
        st.info("👆 Please upload an image or pick a benchmark sample to test the pipelines.")


with tab_analytics:
    st.header("📈 Empirical Benchmark Analytics")
    
    # Load and display CSV results if available
    res_dir = PROJECT_ROOT / "results"
    csv_final = res_dir / "final_comparison.csv"
    csv_cond = res_dir / "condition_wise_comparison.csv"
    csv_ablation = res_dir / "ablation_results.csv"

    if csv_final.exists():
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader("1. Primary Pipeline Comparison")
            df_f = pd.read_csv(csv_final)
            st.dataframe(df_f, use_container_width=True)
        with col_t2:
            st.subheader("2. Ablation Study Across Pipelines")
            df_ab = pd.read_csv(csv_ablation)
            st.dataframe(df_ab, use_container_width=True)

        st.subheader("3. Condition-Wise Evaluation Table")
        df_c = pd.read_csv(csv_cond)
        st.dataframe(df_c, use_container_width=True)

    # Display Research Plots
    st.subheader("4. Research Visualizations & Curves")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        if (res_dir / "condition_wise_map.png").exists():
            st.image(str(res_dir / "condition_wise_map.png"), caption="Condition-Wise mAP@50 Comparison", use_container_width=True)
        if (res_dir / "condition_training_curve.png").exists():
            st.image(str(res_dir / "condition_training_curve.png"), caption="PyTorch Condition Classifier Training Curves", use_container_width=True)

    with p_col2:
        if (res_dir / "mAP50_comparison.png").exists():
            st.image(str(res_dir / "mAP50_comparison.png"), caption="Overall mAP@50 Comparison Across Pipelines", use_container_width=True)
        if (res_dir / "condition_confusion_matrix.png").exists():
            st.image(str(res_dir / "condition_confusion_matrix.png"), caption="Condition Classifier Confusion Matrix", use_container_width=True)


with tab_about:
    st.header("📖 Research Methodology & Architecture")
    st.markdown("""
    ### Hypotheses Tested:
    * **H1**: Adaptive image enhancement improves YOLO detection under adverse conditions.
    * **H2**: Condition-specific enhancement outperforms static one-size-fits-all CLAHE.
    * **H3**: Enhancement provides greater benefit under degraded conditions than clear conditions.
    * **H4**: Downstream object detection depends on task-oriented filtering rather than subjective visual aesthetics.

    ### Environmental Condition Enhancers:
    * **Clear**: Identity pass-through (preserves original features without injecting artifacts).
    * **Rain**: Classical bilateral filtering + high-boost unsharp masking + contrast stretching.
    * **Fog**: Dark Channel Prior (DCP) dehazing + guided transmission refinement.
    * **Night**: Nonlinear power-law Gamma expansion ($\gamma=0.60$) + luminance CLAHE.
    * **Dawn/Dusk**: Transition lighting illumination balancing ($\gamma=0.85$) + midtone contrast adjustment.

    ### Hardware & Execution:
    * Detector: `yolo11n.pt` (Ultralytics)
    * Classifier: `ResNet-18` (5 classes)
    * Dataset: Real BDD100K benchmark images
    """)
