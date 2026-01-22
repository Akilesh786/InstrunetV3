import streamlit as st
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
import joblib
import tempfile
import os
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Instrunet AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# GLOBAL CSS (UI POLISH)
# =========================
st.markdown("""
<style>
body { background-color: #0e1117; }
h1, h2, h3 { color: #ffffff; }
.stButton>button {
    border-radius: 10px;
    padding: 10px 24px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = "Upload Audio"

if "history" not in st.session_state:
    st.session_state.history = []

if "current_result" not in st.session_state:
    st.session_state.current_result = None

# =========================
# MODEL PATHS
# =========================
MODEL_PATH = "models/irmas_instrument_only_cnn_final.keras"
LABEL_PATH = "models/label_encoder_instrument_only.pkl"

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@st.cache_resource
def load_label_encoder():
    return joblib.load(LABEL_PATH)

model = load_model()
label_encoder = load_label_encoder()

# =========================
# INSTRUMENT → FAMILY MAP
# =========================
INSTRUMENT_TO_FAMILY = {
    "cel": "strings", "gac": "strings", "gel": "strings", "vio": "strings",
    "cla": "woodwind", "flu": "woodwind", "sax": "woodwind",
    "tru": "brass",
    "pia": "keyboard", "org": "keyboard"
}

FAMILY_INSTRUMENTS = {
    "strings": ["cel", "gac", "gel", "vio"],
    "woodwind": ["cla", "flu", "sax"],
    "brass": ["tru"],
    "keyboard": ["pia", "org"]
}

# =========================
# FEATURE EXTRACTION
# =========================
def extract_mfcc(path, duration=3, sr=22050, n_mfcc=40, max_len=130):
    y, sr = librosa.load(path, sr=sr, duration=duration)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
    if mfcc.shape[0] < max_len:
        mfcc = np.pad(mfcc, ((0, max_len - mfcc.shape[0]), (0, 0)))
    return mfcc[:max_len]

# =========================
# AUDIO VISUALIZATION
# =========================
def plot_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050, duration=3)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # Waveform
    axs[0].plot(y)
    axs[0].set_title("Waveform")

    # Spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=axs[1])
    axs[1].set_title("Mel Spectrogram")
    fig.colorbar(img, ax=axs[1])

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    img2 = librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=axs[2])
    axs[2].set_title("MFCC")
    fig.colorbar(img2, ax=axs[2])

    plt.tight_layout()
    return fig

# =========================
# INSTRUMENT DISTRIBUTION (MODEL-BASED)
# =========================
def score_instruments(pred_raw, family):
    instruments = FAMILY_INSTRUMENTS[family]
    scores = {}
    total = 0.0

    for inst in instruments:
        idx = list(label_encoder.classes_).index(inst)
        val = float(pred_raw[idx])
        scores[inst] = val
        total += val

    return {k: round((v / total) * 100, 2) for k, v in scores.items()}

# =========================
# MAIN ANALYSIS
# =========================
def analyze_audio(audio_path):
    mfcc = extract_mfcc(audio_path)
    X = mfcc[np.newaxis, ..., np.newaxis]

    pred = model.predict(X)[0]
    idx = np.argmax(pred)
    confidence = float(np.max(pred))

    instrument = label_encoder.inverse_transform([idx])[0]
    family = INSTRUMENT_TO_FAMILY[instrument]

    distribution = score_instruments(pred, family)

    return {
        "family": family,
        "instrument": instrument,
        "confidence": confidence,
        "distribution": distribution,
        "audio_path": audio_path
    }

# =========================
# SIDEBAR NAVIGATION
# =========================
with st.sidebar:
    st.title("🎼 Instrunet AI")
    st.session_state.page = st.radio(
        "Navigation",
        [
            "Upload Audio",
            "Latest Prediction",
            "Instrument Distribution",
            "Audio Analysis",
            "History",
            "About"
        ]
    )

# =========================
# PAGE: UPLOAD AUDIO
# =========================
if st.session_state.page == "Upload Audio":
    st.header("📤 Upload Audio")
    uploaded = st.file_uploader("Upload WAV or MP3", type=["wav", "mp3"])

    if uploaded:
        st.audio(uploaded)

        if st.button("Analyze Audio"):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                path = tmp.name

            result = analyze_audio(path)
            result["filename"] = uploaded.name

            st.session_state.current_result = result
            st.session_state.history.insert(0, result)

            st.success("Analysis complete!")
            st.session_state.page = "Latest Prediction"

# =========================
# PAGE: LATEST PREDICTION
# =========================
if st.session_state.page == "Latest Prediction":
    res = st.session_state.current_result
    if not res:
        st.warning("No audio analyzed yet.")
    else:
        st.markdown("## 🎵 Family Prediction")
        st.markdown(f"### **{res['family'].upper()}**")
        st.metric("Model Confidence", f"{res['confidence']*100:.2f}%")

# =========================
# PAGE: INSTRUMENT DISTRIBUTION
# =========================
if st.session_state.page == "Instrument Distribution":
    res = st.session_state.current_result
    if not res:
        st.warning("Analyze audio first.")
    else:
        st.markdown("## 📊 Instrument Distribution")
        for inst, pct in res["distribution"].items():
            st.markdown(f"""
            <div style="margin-bottom:12px;">
            <b>{inst.upper()}</b>
            <div style="background:#222;border-radius:8px;">
            <div style="width:{pct}%;background:#1f77b4;padding:6px;border-radius:8px;color:white;">
            {pct}%
            </div></div></div>
            """, unsafe_allow_html=True)

# =========================
# PAGE: AUDIO ANALYSIS
# =========================
if st.session_state.page == "Audio Analysis":
    res = st.session_state.current_result
    if not res:
        st.warning("Analyze audio first.")
    else:
        st.markdown("## 📈 Audio Feature Analysis")
        fig = plot_audio_features(res["audio_path"])
        st.pyplot(fig)

# =========================
# PAGE: HISTORY
# =========================
if st.session_state.page == "History":
    st.markdown("## 🕘 History")
    if not st.session_state.history:
        st.info("No history yet.")
    else:
        for i, item in enumerate(st.session_state.history):
            if st.button(f"{item['filename']} → {item['family']}", key=i):
                st.session_state.current_result = item
                st.session_state.page = "Latest Prediction"

# =========================
# PAGE: ABOUT
# =========================
if st.session_state.page == "About":
    st.markdown("""
    ## ℹ️ Instrunet AI  
    **Two-stage music instrument recognition system**

    - Dataset: IRMAS (reduced)
    - Features: MFCC
    - Model: CNN
    - Interpretation: Family-based probability breakdown
    - Visualization: Waveform, Spectrogram, MFCC
    """)