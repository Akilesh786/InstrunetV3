import streamlit as st
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Instrunet AI V2", 
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# DYNAMIC PATH HANDLING 
# =========================
# This ensures the app finds the model whether on Windows, Mac, or Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# If your model is inside a 'models' folder, use this:
MODEL_PATH = os.path.join(BASE_DIR, "models", "irmas_instrument_model.h5")
# If your model is in the main folder, use: MODEL_PATH = os.path.join(BASE_DIR, "irmas_instrument_model.h5")

# =========================
# MODEL & LABELS
# =========================
INSTRUMENTS = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']

@st.cache_resource
def load_new_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}. Please check your folder structure!")
        return None
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_new_model()

# =========================
# FAMILY MAP
# =========================
FAMILY_MAP = {
    "strings": ["cel", "gac", "gel", "vio"],
    "woodwind": ["cla", "flu", "sax"],
    "brass": ["tru"],
    "keyboard": ["pia", "org"],
    "voice": ["voi"]
}

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
# FEATURE EXTRACTION
# =========================
def extract_v2_features(path):
    # Load 3 seconds to match training
    y, sr = librosa.load(path, sr=22050, duration=3)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T

    # Pad or truncate to exactly 130 time steps
    if mfcc.shape[0] < 130:
        mfcc = np.pad(mfcc, ((0, 130 - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:130]

    return mfcc, y, sr

# =========================
# MAIN ANALYSIS
# =========================
def analyze_v2(audio_path):
    mfcc, y, sr = extract_v2_features(audio_path)
    # Reshape for CNN input: (batch, time, features, channels)
    X = mfcc.reshape(1, 130, 40, 1)

    raw_preds = model.predict(X, verbose=0)[0]
    instr_scores = {INSTRUMENTS[i]: float(raw_preds[i]) for i in range(len(INSTRUMENTS))}

    # Family logic: Take the max confidence found within that family
    family_scores = {}
    for fam, members in FAMILY_MAP.items():
        family_scores[fam] = max([instr_scores[m] for m in members])

    top_family = max(family_scores, key=family_scores.get)
    top_instrument = max(instr_scores, key=instr_scores.get)

    return {
        "family": top_family,
        "instrument": top_instrument,
        "confidence": family_scores[top_family],
        "distribution": instr_scores,
        "audio_path": audio_path,
        "raw_y": y,
        "sr": sr,
        "filename": os.path.basename(audio_path)
    }

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("🎼 Instrunet AI V2")
    st.session_state.page = st.radio(
        "Navigation",
        ["Upload Audio", "Latest Prediction", "Instrument Distribution", "Audio Analysis", "History", "About"]
    )
    st.divider()
    st.info("Powered by Multi-label CNN")

# =========================
# PAGES
# =========================
if st.session_state.page == "Upload Audio":
    st.header("📤 Upload Audio")
    uploaded = st.file_uploader("Upload WAV or MP3 (3-second clips work best)", type=["wav", "mp3"])

    if uploaded:
        st.audio(uploaded)
        if st.button("Analyze Audio", use_container_width=True):
            if model is None:
                st.error("Model not loaded. Fix path issues first.")
            else:
                with st.spinner("Analyzing spectral patterns..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(uploaded.getvalue())
                        path = tmp.name

                    result = analyze_v2(path)
                    result["filename"] = uploaded.name

                    st.session_state.current_result = result
                    st.session_state.history.insert(0, result)
                    st.success("Analysis complete!")
                    st.session_state.page = "Latest Prediction"
                    st.rerun()

elif st.session_state.page == "Latest Prediction":
    res = st.session_state.current_result
    if not res:
        st.warning("No audio analyzed yet.")
    else:
        st.markdown(f"## 🎵 Predicted Family: **{res['family'].upper()}**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Family Confidence", f"{res['confidence']*100:.2f}%")
        with col2:
            st.metric("Top Instrument", res['instrument'].upper())
        st.info(f"The model identifies this as part of the **{res['family']}** family.")

elif st.session_state.page == "Instrument Distribution":
    res = st.session_state.current_result
    if not res:
        st.warning("Analyze audio first.")
    else:
        st.markdown(f"## 📊 {res['family'].upper()} Instrument Breakdown")
        st.write("Specific confidence levels for instruments in this family:")
        for inst in FAMILY_MAP[res['family']]:
            val = res["distribution"][inst]
            st.write(f"**{inst.upper()}**")
            st.progress(val)
            st.caption(f"Confidence: {val*100:.2f}%")

elif st.session_state.page == "Audio Analysis":
    res = st.session_state.current_result
    if not res:
        st.warning("Analyze audio first.")
    else:
        st.markdown("## 📈 Audio Visualization")
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))

        # Waveform
        librosa.display.waveshow(res['raw_y'], sr=res['sr'], ax=ax[0], color="#1f77b4")
        ax[0].set_title("Waveform (Time Domain)")

        # Spectrogram
        S = librosa.feature.melspectrogram(y=res['raw_y'], sr=res['sr'])
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel', ax=ax[1], sr=res['sr'])
        ax[1].set_title("Mel Spectrogram (Frequency Domain)")
        fig.colorbar(img, ax=ax[1], format="%+2.0f dB")

        plt.tight_layout()
        st.pyplot(fig)

elif st.session_state.page == "History":
    st.markdown("## 🕘 History")
    if not st.session_state.history:
        st.info("No history yet.")
    else:
        for i, item in enumerate(st.session_state.history[:5]): # Show last 5
            if st.button(f"{item['filename']} → {item['family'].upper()} ({item['confidence']*100:.1f}%)", key=f"hist_{i}"):
                st.session_state.current_result = item
                st.session_state.page = "Latest Prediction"
                st.rerun()

elif st.session_state.page == "About":
    st.markdown("""
    ## ℹ️ Instrunet AI V2  
    **Multi-label Music Instrument Recognition**

    This version uses a **Convolutional Neural Network (CNN)** trained on the IRMAS dataset. Unlike standard classifiers, this model uses **Sigmoid activation** to detect multiple instruments simultaneously.

    - **Input:** 3-second audio clip  
    - **Features:** 40 Mel-Frequency Cepstral Coefficients (MFCCs)  
    - **Architecture:** 4-Layer CNN  
    - **Classes:** 11 Instruments (Cello, Flute, Guitar, Piano, Sax, etc.)
    """)