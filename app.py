import os
import io
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import densenet121

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
MODEL_PATH = "best.pt"
HF_MODEL_URL = "https://huggingface.co/eseeyuh/pneumonia-xai-detector/resolve/main/best.pt"

# Auto-download model if not present
if not os.path.exists(MODEL_PATH):
    import urllib.request
    with st.spinner("Downloading model weights..."):
        urllib.request.urlretrieve(HF_MODEL_URL, MODEL_PATH)  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

C_INK = "#16202b"
C_NORMAL = "#1f9d8b"
C_PNEU = "#d1495b"
C_AMBER = "#e0a458"
C_MUTED = "#5c6b7a"

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    model = densenet121(weights=None)           
    in_feats = model.classifier.in_features
    model.classifier = nn.Sequential(           
        nn.Dropout(p=0.3),
        nn.Linear(in_feats, 2),
    )
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def preprocess(pil_img: Image.Image):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    rgb = np.array(img).astype(np.float32) / 255.0
    tensor = eval_tfm(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    return tensor, rgb


def predict(model, tensor):
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return int(probs.argmax()), probs


def grad_cam_overlay(model, tensor, rgb, pred_idx):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    target_layers = [model.features[-1]]       
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale = cam(
            input_tensor=tensor,
            targets=[ClassifierOutputTarget(pred_idx)],
        )[0]
    overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
    return overlay                            


def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout(model, tensor, n_passes: int):
    model.eval()
    enable_dropout(model)
    samples = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(tensor)
            samples.append(torch.softmax(logits, dim=1).cpu().numpy()[0])
    model.eval()                                
    samples = np.stack(samples, axis=0)       

    mean_probs = samples.mean(axis=0)
    eps = 1e-12
    entropy = float(-np.sum(mean_probs * np.log(mean_probs + eps)))
    variance = float(samples[:, 1].var())       
    return {
        "samples": samples,
        "mean_probs": mean_probs,
        "pred_idx": int(mean_probs.argmax()),
        "entropy": entropy,        
        "variance": variance,
        "pneu_samples": samples[:, 1],
    }


def card(label, value, color=C_INK, sub=""):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value" style="color:{color}">{value}</div>
      {sub_html}
    </div>
    """


def entropy_label(entropy):
    if entropy < 0.20:
        return "Low uncertainty", C_NORMAL
    if entropy < 0.45:
        return "Moderate uncertainty", C_AMBER
    return "High uncertainty", C_PNEU


def uncertainty_plot(pneu_samples):
    fig, ax = plt.subplots(figsize=(6, 2.6), dpi=120)
    ax.hist(pneu_samples, bins=15, range=(0, 1),
            color=C_PNEU, alpha=0.75, edgecolor="white")
    ax.axvline(0.5, color=C_MUTED, ls="--", lw=1)
    ax.axvline(pneu_samples.mean(), color=C_INK, lw=2,
               label=f"mean = {pneu_samples.mean():.2f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(pneumonia) per stochastic pass", fontsize=9)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


st.set_page_config(page_title="XAI Pneumonia Detector",
                layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

.stApp { background: #f4f6f8; }
html, body, [class*="css"], .stMarkdown, p, span, label, div {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
}
.block-container { padding-top: 2.2rem; max-width: 1150px; }

.app-title {
    font-size: 2.0rem; font-weight: 700; color: #16202b;
    letter-spacing: -0.02em; margin-bottom: .1rem;
}
.app-sub { color: #5c6b7a; font-size: .98rem; margin-bottom: 1.4rem; }
.rule { height: 3px; width: 56px; background: #0e7c86;
        border-radius: 3px; margin: .2rem 0 1.3rem 0; }

.metric-card {
    background: #ffffff; border: 1px solid #e3e8ec; border-radius: 14px;
    padding: 18px 20px; box-shadow: 0 1px 2px rgba(22,32,43,.04);
    height: 100%;
}
.metric-label {
    font-size: .74rem; letter-spacing: .08em; text-transform: uppercase;
    color: #5c6b7a; font-weight: 600; margin-bottom: 6px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.9rem; font-weight: 600;
    line-height: 1.1;
}
.metric-sub { font-size: .82rem; color: #5c6b7a; margin-top: 4px; }

.verdict {
    border-radius: 14px; padding: 22px 26px; color: #fff;
    box-shadow: 0 6px 18px rgba(22,32,43,.10);
}
.verdict-label { font-size: .8rem; letter-spacing: .1em; text-transform: uppercase;
                 opacity: .85; font-weight: 600; }
.verdict-class { font-size: 2.3rem; font-weight: 700; letter-spacing: -.02em; }

.section-h {
    font-size: 1.15rem; font-weight: 600; color: #16202b;
    margin: 1.8rem 0 .2rem 0;
}
.disclaimer {
    background: #fff6e9; border: 1px solid #f0d9b5; border-radius: 12px;
    padding: 12px 16px; font-size: .85rem; color: #7a5a23; margin-top: 1.5rem;
}
[data-testid="stImage"] img { border-radius: 12px; border: 1px solid #e3e8ec; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="app-title">Explainable-AI Pneumonia Detector</div>',
            unsafe_allow_html=True)
st.markdown('<div class="app-sub">DenseNet-121 · chest X-ray classification '
            'with Grad-CAM and MC-Dropout uncertainty</div>',
            unsafe_allow_html=True)
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### Settings")
    n_passes = st.slider("MC-Dropout passes (T)", 5, 60, 30, step=5,
                         help="More passes = smoother uncertainty estimate, "
                              "but slower. The notebook used 30.")
    show_cam = st.checkbox("Compute Grad-CAM heatmap", value=True)
    st.markdown("---")
    st.caption(f"Device: **{DEVICE.type.upper()}**")
    st.caption(f"Weights file: `{MODEL_PATH}`")

try:
    model = load_model(MODEL_PATH)
    model_ok = True
except FileNotFoundError:
    model_ok = False
    st.error(
        f"Model weights `{MODEL_PATH}` not found.\n\n"
        "Run the notebook through the training cell, download "
        "`/content/results/models/best.pt`, and place it in the same "
        "folder as `app.py`. See the README for the exact steps."
    )
except Exception as e:
    model_ok = False
    st.error(f"Could not load the model: {e}")

uploaded = st.file_uploader(
    "Upload a chest X-ray (JPEG / PNG)",
    type=["jpg", "jpeg", "png"],
    disabled=not model_ok,
)

if uploaded and model_ok:
    pil_img = Image.open(io.BytesIO(uploaded.read()))
    tensor, rgb = preprocess(pil_img)

    with st.spinner("Running model…"):
        pred_idx, probs = predict(model, tensor)
        mc = mc_dropout(model, tensor, n_passes)
        overlay = grad_cam_overlay(model, tensor, rgb, pred_idx) if show_cam else None

    pred_name = CLASS_NAMES[pred_idx]
    pred_color = C_PNEU if pred_idx == 1 else C_NORMAL
    confidence = float(probs[pred_idx])
    ent_text, ent_color = entropy_label(mc["entropy"])

    st.markdown(
        f"""
        <div class="verdict" style="background:{pred_color}">
          <div class="verdict-label">Predicted class</div>
          <div class="verdict-class">{pred_name}</div>
          <div style="opacity:.9">deterministic confidence: {confidence*100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-h">Prediction</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.markdown(card("P(normal)", f"{probs[0]*100:.1f}%", C_NORMAL),
                unsafe_allow_html=True)
    m2.markdown(card("P(pneumonia)", f"{probs[1]*100:.1f}%", C_PNEU),
                unsafe_allow_html=True)
    m3.markdown(card("MC mean P(pneu)", f"{mc['mean_probs'][1]*100:.1f}%", C_INK,
                     sub=f"over {n_passes} passes"),
                unsafe_allow_html=True)

    st.markdown('<div class="section-h">Where the model looked</div>',
                unsafe_allow_html=True)
    if show_cam:
        c1, c2 = st.columns(2)
        c1.image((rgb * 255).astype(np.uint8), caption="Input (224×224)",
                 use_container_width=True)
        c2.image(overlay, caption="Grad-CAM — warm = most influential regions",
                 use_container_width=True)
    else:
        st.image((rgb * 255).astype(np.uint8), caption="Input (224×224)", width=360)
        st.caption("Grad-CAM disabled in the sidebar.")

    st.markdown('<div class="section-h">Uncertainty (MC-Dropout)</div>',
                unsafe_allow_html=True)
    u1, u2, u3 = st.columns(3)
    u1.markdown(card("Predictive entropy", f"{mc['entropy']:.3f}", ent_color,
                     sub="0 = certain · 0.693 = max"),
                unsafe_allow_html=True)
    u2.markdown(card("Confidence band", ent_text, ent_color),
                unsafe_allow_html=True)
    u3.markdown(card("Variance of P(pneu)", f"{mc['variance']:.4f}", C_INK,
                     sub="spread across passes"),
                unsafe_allow_html=True)

    st.pyplot(uncertainty_plot(mc["pneu_samples"]), use_container_width=True)
    st.caption(
        "Each pass keeps Dropout active, so the model gives a slightly different "
        "answer every time. A wide spread or high entropy means the model is "
        "unsure — a good signal that a radiologist should take a closer look."
    )

elif model_ok:
    st.info("Upload a chest X-ray image to get a prediction, a Grad-CAM "
            "heatmap, and an uncertainty estimate.")
