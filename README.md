# XAI Pneumonia Detector

A Streamlit web app that wraps a trained DenseNet-121 model for pneumonia detection in chest X-rays. Upload a scan and get three things back: a classification, a Grad-CAM heatmap showing where the model looked, and an MC-Dropout uncertainty estimate.

> **Note:** The app only runs inference — it does not train the model. You need `best.pt` from the training notebook before you can use it.

---

## What it does

- Classifies a chest X-ray as **NORMAL** or **PNEUMONIA**
- Shows class probabilities from a standard forward pass
- Overlays a **Grad-CAM** heatmap on the input image
- Runs **T stochastic forward passes** (MC-Dropout) to estimate prediction uncertainty via entropy and variance

---

## Results

| Metric | This model | Kermany et al. baseline |
|--------|-----------|------------------------|
| AUC-ROC | **0.9975** | 0.968 |
| Accuracy | **0.9676** | 0.928 |
| F1 | **0.9774** | 0.916 |
| Precision | **0.9976** | 0.901 |

Deferring the 30% most uncertain cases (by predictive entropy) raises retained accuracy to **0.998**. At 40% deferral the retained set hits 100%.

---

## Research paper

**Explainable Pneumonia Triage from Chest X-Rays**  
Daryn Shaidarov — University of Portsmouth, 2025  
Supervised by Dr Alexander Gegov, Reader in Explainable AI

*arXiv preprint — link coming soon*

---

## Setup

### 1. Download the model weights

```bash
wget https://huggingface.co/eseeyuh/pneumonia-xai-detector/resolve/main/best.pt
```

Or download manually from
[Hugging Face](https://huggingface.co/eseeyuh/pneumonia-xai-detector)
and place `best.pt` in the root directory.

### 2. Put everything in one folder

```
project/
├── app.py
├── requirements.txt
├── README.md
└── best.pt
```

`best.pt` needs to sit next to `app.py` — the path is hardcoded.

### 3. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First install takes a few minutes because of PyTorch.

### 4. Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Usage

Upload a `.jpg` or `.png` chest X-ray using the file uploader. Results appear in a few seconds.

**Reading the output:**
- **Probabilities** — single deterministic forward pass
- **Grad-CAM** — warm (red) regions had the most influence on the prediction. For true pneumonia cases these should land on the lung fields, not on image borders or scanner annotations
- **Entropy** — 0 means the model is certain, 0.693 is maximum uncertainty for a binary classifier. A wide histogram spread means the model is unsure and the case probably warrants a second look

You can adjust the number of MC-Dropout passes (T) and toggle Grad-CAM on/off in the sidebar.

---

## Troubleshooting

**`Model weights best.pt not found`** — `best.pt` is not in the same folder as `app.py`. Go back to step 1.

**`load_state_dict` key mismatch** — the weights are from a different architecture. Make sure `best.pt` comes from this notebook specifically (DenseNet-121 with `Dropout(0.3) → Linear(..., 2)` head).

**Grad-CAM import error** — run `pip install --upgrade grad-cam opencv-python-headless`, or just uncheck "Compute Grad-CAM" in the sidebar. Predictions and uncertainty still work without it.

**No GPU** — fine, the app runs on CPU. It'll be a bit slower but works the same. The sidebar shows the active device.

---

## Model details

| Parameter | Value |
|-----------|-------|
| Architecture | DenseNet-121, head: `Dropout(0.3) + Linear → 2` |
| Input | 224×224 RGB, ImageNet normalisation |
| Grad-CAM target layer | `model.features[-1]` |
| MC-Dropout | T passes, only Dropout layers active |
| Uncertainty metrics | Predictive entropy, variance of P(pneumonia) |

---

## Tech stack

Python · PyTorch · Streamlit · pytorch-grad-cam · NumPy · Pillow · Matplotlib
