import numpy as np
import torch
import librosa
import joblib
import json
from pathlib import Path
from transformers import AutoFeatureExtractor, HubertModel
import pandas as pd
import gradio as gr

# === Load Config & Artifacts ===
MODEL_DIR = Path("models")
cfg = json.load(open(MODEL_DIR / "config.json"))
BEST_LAYER = cfg["best_layer"]
SR = cfg["sample_rate"]
MODEL_NAME = cfg["hubert_model"]

extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
hubert = HubertModel.from_pretrained(MODEL_NAME).eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
hubert.to(device)

clf = joblib.load(MODEL_DIR / "accent_clf_latest.pkl")
scaler = joblib.load(MODEL_DIR / "accent_scaler_latest.pkl")
le = joblib.load(MODEL_DIR / "accent_label_encoder_latest.pkl")
labels = list(le.classes_)

ACCENT_TO_CUISINE = {
    "andhra_pradesh": ["Gongura Pachadi", "Pesarattu", "Kodi Kura"],
    "kerala": ["Appam", "Puttu", "Avial", "Fish Moilee"],
    "karnataka": ["Bisi Bele Bath", "Ragi Mudde", "Mysore Masala Dosa"],
    "tamil": ["Idli", "Pongal", "Chettinad Chicken"],
    "jharkhand": ["Thekua", "Rugra", "Dhuska"],
    "gujrat": ["Undhiyu", "Dhokla", "Khandvi"],
}

def _to_float_mono(y: np.ndarray) -> np.ndarray:
    """Ensure float32 in [-1, 1] and mono."""
    # If stereo: average channels
    if y.ndim == 2:
        y = np.mean(y, axis=1)
    # Convert ints to float32 in [-1,1]
    if not np.issubdtype(y.dtype, np.floating):
        # int16/24/32 etc.
        maxv = np.iinfo(y.dtype).max
        y = y.astype(np.float32) / maxv
    else:
        y = y.astype(np.float32, copy=False)
    # Avoid all-zeros dividing later
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y

@torch.inference_mode()
def hubert_mean_vec(y: np.ndarray, sr_in: int, target_layer: int, max_sec: float = 12.0):
    y = _to_float_mono(y)
    # Resample if needed
    if sr_in != SR:
        y = librosa.resample(y, orig_sr=sr_in, target_sr=SR)
    # Trim
    if y.size > int(max_sec * SR):
        y = y[: int(max_sec * SR)]
    # Tokenize + forward
    inputs = extractor(y, sampling_rate=SR, return_tensors="pt").to(device)
    out = hubert(**inputs, output_hidden_states=True)
    hs = out.hidden_states[target_layer]  # [1, T, H]
    vec = torch.mean(hs, dim=1).squeeze(0).cpu().numpy().astype("float32")  # (H,)
    return vec


def _run_pipeline(audio_tuple):
    if audio_tuple is None:
        return None
    sr_in, y = audio_tuple
    if y is None or len(y) == 0:
        return None

    vec = hubert_mean_vec(y, sr_in=sr_in, target_layer=BEST_LAYER)
    X = scaler.transform([vec])

    # Try predict_proba (LR etc.). If not available (LinearSVC), fall back to decision_function → softmax.
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
    else:
        # decision scores to pseudo-probabilities (temperature=1 softmax)
        scores = clf.decision_function(X)
        scores = np.array(scores, dtype=np.float64).reshape(-1)
        m = scores.max()
        exps = np.exp(scores - m)
        proba = (exps / exps.sum())

    top_idx = int(np.argmax(proba))
    accent = le.inverse_transform([top_idx])[0]
    dishes = ACCENT_TO_CUISINE.get(accent, [])
    probs = sorted([(labels[i], float(proba[i])) for i in range(len(labels))],
                   key=lambda x: x[1], reverse=True)
    return accent, dishes, probs, float(proba[top_idx])

def predict(mic_audio, file_audio, min_conf=0.0):
    # Prefer mic if present, otherwise use file
    result = _run_pipeline(mic_audio) or _run_pipeline(file_audio)
    if result is None:
        return "No audio detected", [], [], ""
    accent, dishes, probs, conf = result
    if conf < min_conf:
        return f"Low confidence ({conf:.2f}). Please re-record closer to mic.", [], probs, f"{conf:.2f}"
    return accent, dishes, probs, f"{conf:.2f}"

# === Gradio UI ===
THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    body_background_fill="linear-gradient(180deg, #0f172a 0%, #0b1220 100%)",
    block_background_fill="rgba(30,41,59,0.6)",
    border_color_primary="#334155",
)

def _probs_to_df(probs):
    # probs: list[(label, prob), ...] descending
    if not probs:
        return pd.DataFrame({"Accent": [], "Probability": []})
    labels, ps = zip(*probs)
    return pd.DataFrame({"Accent": labels, "Probability": ps})

def _predict_ui(mic_audio, file_audio, min_conf=0.0):
    """
    Return types must match the components:
      accent_out: str
      dishes_out: JSON-serializable (list)
      probs_plot: DataFrame
      conf_out: str
      notes_out: list[tuple[str, str|None]]  (for HighlightedText)
    """
    res = predict(mic_audio, file_audio, min_conf)

    # Your predict() returns either:
    #   ("No audio detected", [], [], "")    OR
    #   (accent:str, dishes:list, probs:list[(label,prob)], conf:str)
    if not isinstance(res, tuple) or len(res) != 4:
        return "Error", [], _probs_to_df([]), "", [("Unexpected output shape", "error")]

    accent, dishes, probs, conf = res

    # Normalize confidence to string for the textbox
    try:
        conf_f = float(conf)
        conf_str = f"{conf_f:.2f}"
    except Exception:
        conf_f = None
        conf_str = str(conf) if conf is not None else ""

    # Build notes for HighlightedText: list of (text, category) tuples
    notes = []
    if isinstance(accent, str) and accent.lower().startswith("low confidence"):
        notes.append((accent, "warning"))
    if conf_f is not None:
        notes.append((f"Top-1 confidence: {conf_f:.2f}", "info"))

    # If no audio path (your function returns a message in 'accent')
    if accent == "No audio detected":
        return accent, [], _probs_to_df([]), "", [(accent, "error")]

    df = _probs_to_df(probs)
    return accent, dishes, df, conf_str, notes

with gr.Blocks(theme=THEME, title="Accent-Aware Cuisine Recommender") as demo:
    gr.HTML("""
    <style>
      .aa-card  { border:1px solid #334155; background:rgba(15,23,42,0.55); padding:16px; border-radius:18px; }
    </style>
    """)

    gr.Markdown(
        "# 🎙️ Accent-Aware Cuisine Recommender\n"
        "Detect the accent in a short English utterance (2–5s) using **HuBERT** embeddings "
        "+ a lightweight classifier, then suggest regional dishes. "
        "_Tip: record close to the mic for best results._"
    )

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tab("🎤 Use Microphone"):
                mic_input = gr.Audio(sources=["microphone"], type="numpy", label="Record")
            with gr.Tab("📁 Upload File"):
                file_input = gr.Audio(sources=["upload"], type="numpy", label="Audio file")

            min_conf = gr.Slider(0.0, 0.99, value=0.15, step=0.01, label="Minimum confidence (optional)")

            with gr.Row():
                run_btn = gr.Button("Predict", variant="primary")
                clear_btn = gr.Button("Clear")

        with gr.Column(scale=3):
            gr.Markdown("### Results")
            with gr.Row():
                with gr.Column():
                    with gr.Group(elem_classes=["aa-card"]):
                        accent_out = gr.Textbox(label="Predicted Accent", interactive=False)
                        conf_out   = gr.Textbox(label="Top-1 Confidence", interactive=False)
                        notes_out  = gr.HighlightedText(label="Notes", combine_adjacent=True)
                with gr.Column():
                    with gr.Group(elem_classes=["aa-card"]):
                        dishes_out = gr.JSON(label="🍽️ Recommended Dishes")

            with gr.Group(elem_classes=["aa-card"]):
                probs_plot = gr.BarPlot(
                    value=None, x="Accent", y="Probability",
                    title="Class Probabilities", height=420, y_lim=(0,1.0)
                )

    def _clear():
        # Must return values matching outputs order & types
        return "", [], _probs_to_df([]), "", []

    run_btn.click(
        fn=_predict_ui,
        inputs=[mic_input, file_input, min_conf],
        outputs=[accent_out, dishes_out, probs_plot, conf_out, notes_out]
    )
    clear_btn.click(_clear, outputs=[accent_out, dishes_out, probs_plot, conf_out, notes_out])
demo.launch(share=True)