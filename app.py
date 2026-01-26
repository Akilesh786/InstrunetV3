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
# PATHS & MODEL LOADING
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure your model is in the 'models' folder
MODEL_PATH = os.path.join(BASE_DIR, "models", "irmas_instrument_model.h5")
INSTRUMENTS = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']

@st.cache_resource
def load_new_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    else:
        st.error(f"Model not found at {MODEL_PATH}. Check your folder structure!")
        return None

model = load_new_model()

FAMILY_MAP = {
    "strings": ["cel", "gac", "gel", "vio"],
    "woodwind": ["cla", "flu", "sax"],
    "brass": ["tru"],
    "keyboard": ["pia", "org"],
    "voice": ["voi"]
}

# =========================
# INITIALIZE SESSION STATE
# =========================
if "page" not in st.session_state: 
    st.session_state.page = "About"
if "history" not in st.session_state: 
    st.session_state.history = []
if "current_result" not in st.session_state: 
    st.session_state.current_result = None

# =========================
# CORE FUNCTIONS
# =========================
def analyze_v2(audio_path):
    # Load 3s audio as MFCC (130, 40)
    y, sr = librosa.load(audio_path, sr=22050, duration=3)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
    
    # Pad or truncate to 130 time steps
    if mfcc.shape[0] < 130:
        mfcc = np.pad(mfcc, ((0, 130 - mfcc.shape[0]), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:130]
        
    X = mfcc.reshape(1, 130, 40, 1)
    
    # Predict
    raw_preds = model.predict(X, verbose=0)[0]
    instr_scores = {INSTRUMENTS[i]: float(raw_preds[i]) for i in range(len(INSTRUMENTS))}
    
    # Family scores based on max instrument confidence
    family_scores = {fam: max([instr_scores[m] for m in members]) for fam, members in FAMILY_MAP.items()}
    top_family = max(family_scores, key=family_scores.get)
    
    return {
        "family": top_family,
        "instrument": max(instr_scores, key=instr_scores.get),
        "confidence": family_scores[top_family],
        "distribution": instr_scores,
        "raw_y": y, 
        "sr": sr
    }

# =========================
# SIDEBAR NAVIGATION
# =========================
with st.sidebar:
    st.title("🎼 Instrunet AI V2")
    st.markdown("---")
    
    st.subheader("📌 Navigation")
    # Mapping current page to radio index
    pages = ["About", "Upload & Analyze", "Instrument Distribution", "Audio Analysis", "History"]
    current_index = pages.index(st.session_state.page)
    
    choice = st.radio("Go to:", pages, index=current_index)
    
    # Only update page if the user clicked the radio (prevents infinite loop with rerun)
    if choice != st.session_state.page:
        st.session_state.page = choice
        st.rerun()
    
    st.markdown("---")
    if st.session_state.history:
        st.subheader("📜 Recent History")
        for i, h in enumerate(st.session_state.history[:3]):
            if st.button(f"{h['filename'][:12]}.. ({h['family']})", key=f"side_{i}"):
                st.session_state.current_result = h
                st.session_state.page = "Instrument Distribution"
                st.rerun()

# =========================
# PAGE ROUTING
# =========================

# PAGE 1: ABOUT (LANDING PAGE)
if st.session_state.page == "About":
    st.header("📖 About Instrunet AI")
    st.write("""
    Instrunet AI is a state-of-the-art instrument recognition system. 
    Unlike traditional models that only pick one instrument, our **Multi-label CNN** looks for 
    multiple spectral signatures simultaneously.
    """)
    
    

    col1, col2, col3 = st.columns(3)
    col1.metric("Architecture", "4-Layer CNN")
    col2.metric("Dataset", "IRMAS")
    col3.metric("Analysis Window", "3 Seconds")
    
    st.markdown("""
    ### 🛠 How it works:
    1. **Preprocessing:** We convert your audio into a Mel-Spectrogram.
    2. **Analysis:** The CNN detects patterns related to 11 specific instruments.
    3. **Family Logic:** We group detections into Strings, Woodwind, Brass, Keyboard, and Voice.
    """)
    
    if st.button("Get Started 🚀", use_container_width=True):
        st.session_state.page = "Upload & Analyze"
        st.rerun()

# PAGE 2: UPLOAD & ANALYZE
elif st.session_state.page == "Upload & Analyze":
    st.header("📤 Input Audio")
    
    tab1, tab2 = st.tabs(["📁 Upload File", "🎤 Live Record"])
    
    with tab1:
        uploaded = st.file_uploader("Upload a WAV or MP3 clip", type=["wav", "mp3"])
    with tab2:
        recorded = st.audio_input("Record 3 seconds of audio")

    source = uploaded if uploaded else recorded

    if source:
        st.audio(source)
        if st.button("🚀 Analyze Audio", use_container_width=True):
            if model is None:
                st.error("Model not loaded! Check 'models' folder.")
            else:
                with st.spinner("Decoding spectral features..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(source.getvalue())
                        path = tmp.name
                    
                    result = analyze_v2(path)
                    result["filename"] = getattr(source, 'name', 'Live Recording.wav')
                    
                    # Store Result
                    st.session_state.current_result = result
                    st.session_state.history.insert(0, result)
                    
                    # PROGRAMMATIC JUMP
                    st.session_state.page = "Instrument Distribution"
                    st.rerun()

# PAGE 3: DISTRIBUTION (DIRECT REDIRECT TARGET)
elif st.session_state.page == "Instrument Distribution":
    res = st.session_state.current_result
    if not res:
        st.warning("No data found. Please analyze an audio file first!")
        if st.button("Go to Upload"):
            st.session_state.page = "Upload & Analyze"
            st.rerun()
    else:
        st.header(f"📊 Results for: {res['filename']}")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"### Detected Family: \n# {res['family'].upper()}")
            st.metric("Family Confidence", f"{res['confidence']*100:.1f}%")
            st.info(f"Top Instrument: **{res['instrument'].upper()}**")
        
        with c2:
            st.subheader(f"Internal {res['family'].title()} Distribution")
            for inst in FAMILY_MAP[res['family']]:
                val = res["distribution"][inst]
                st.write(f"**{inst.upper()}**")
                st.progress(val)
                st.caption(f"Confidence: {val*100:.1f}%")

# PAGE 4: AUDIO ANALYSIS
elif st.session_state.page == "Audio Analysis":
    res = st.session_state.current_result
    if not res:
        st.warning("Analyze a file to see the spectral breakdown.")
    else:
        st.header("📈 Spectral Visualization")
        
        
        
        fig, ax = plt.subplots(2, 1, figsize=(10, 7))
        librosa.display.waveshow(res['raw_y'], sr=res['sr'], ax=ax[0], color="#2E86C1")
        ax[0].set_title("Time Domain: Waveform")
        
        S = librosa.feature.melspectrogram(y=res['raw_y'], sr=res['sr'])
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', ax=ax[1])
        ax[1].set_title("Frequency Domain: Mel-Spectrogram")
        plt.tight_layout()
        st.pyplot(fig)

# PAGE 5: HISTORY
elif st.session_state.page == "History":
    st.header("📜 Session History")
    if not st.session_state.history:
        st.info("Your analysis history will appear here.")
    else:
        for i, item in enumerate(st.session_state.history):
            with st.expander(f"{item['filename']} — {item['family'].upper()}"):
                col_a, col_b = st.columns(2)
                col_a.write(f"**Top Instrument:** {item['instrument'].upper()}")
                col_a.write(f"**Confidence:** {item['confidence']*100:.1f}%")
                if col_b.button("Reload Report", key=f"hist_btn_{i}"):
                    st.session_state.current_result = item
                    st.session_state.page = "Instrument Distribution"
                    st.rerun()