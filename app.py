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
MODEL_PATH = os.path.join(BASE_DIR, "models", "irmas_instrument_model.h5")
INSTRUMENTS = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']

@st.cache_resource
def load_new_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
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
if "page" not in st.session_state: st.session_state.page = "About"
if "history" not in st.session_state: st.session_state.history = []
if "current_result" not in st.session_state: st.session_state.current_result = None
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I'm your AI Guide. Ask 'How?' to see the spectral data!"}]

# =========================
# CORE FUNCTIONS
# =========================
def analyze_v2(audio_path):
    y, sr = librosa.load(audio_path, sr=22050, duration=3)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
    if mfcc.shape[0] < 130:
        mfcc = np.pad(mfcc, ((0, 130 - mfcc.shape[0]), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:130]
    X = mfcc.reshape(1, 130, 40, 1)
    raw_preds = model.predict(X, verbose=0)[0]
    instr_scores = {INSTRUMENTS[i]: float(raw_preds[i]) for i in range(len(INSTRUMENTS))}
    family_scores = {fam: max([instr_scores[m] for m in members]) for fam, members in FAMILY_MAP.items()}
    top_family = max(family_scores, key=family_scores.get)
    return {
        "family": top_family,
        "instrument": max(instr_scores, key=instr_scores.get),
        "confidence": family_scores[top_family],
        "distribution": instr_scores,
        "raw_y": y, "sr": sr
    }

# =========================
# SIDEBAR NAVIGATION & AI AGENT
# =========================
with st.sidebar:
    st.title("🎼 Instrunet AI V2")
    st.markdown("---")
    
    st.subheader("📌 Navigation")
    pages = ["About", "Upload & Analyze", "Instrument Distribution", "Audio Analysis", "History"]
    choice = st.radio("Go to:", pages, index=pages.index(st.session_state.page))
    
    if choice != st.session_state.page:
        st.session_state.page = choice
        st.rerun()
    
    st.markdown("---")
    
    # --- INTELLIGENT AI AGENT ---
    st.subheader("🤖 AI Technical Guide")
    with st.expander("💬 Chat with Assistant", expanded=True if st.session_state.current_result else False):
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("Ask: 'How did you know?'"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            res = st.session_state.current_result
            if res is None:
                response = "Upload a clip first! I need data to explain the science to you."
            else:
                p = prompt.lower()
                if "how" in p or "why" in p or "spectrogram" in p:
                    response = (f"I identified {res['instrument']} based on spectral peaks. "
                                f"I am switching your view to **Audio Analysis** so you can see the spectrogram evidence.")
                    st.session_state.page = "Audio Analysis"
                elif "confidence" in p or "sure" in p:
                    response = f"My confidence is **{res['confidence']*100:.1f}%**. This is calculated from the MFCC features of the audio."
                else:
                    response = f"The analysis of '{res['filename']}' points to a {res['instrument']}."

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
            if "switching" in response: st.rerun()

        # Clear Chat Button - PLACED UNDER THE CHAT
        st.write("") 
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": "Chat reset! Ready for next analysis."}]
            st.rerun()

# =========================
# PAGE ROUTING
# =========================

if st.session_state.page == "About":
    st.header("📖 About Instrunet AI")
    st.write("A state-of-the-art instrument recognition system using Multi-label CNNs.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Architecture", "4-Layer CNN")
    col2.metric("Dataset", "IRMAS")
    col3.metric("Analysis", "3s Window")
    if st.button("Get Started 🚀", use_container_width=True):
        st.session_state.page = "Upload & Analyze"; st.rerun()

elif st.session_state.page == "Upload & Analyze":
    st.header("📤 Input Audio")
    tab1, tab2 = st.tabs(["📁 Upload File", "🎤 Live Record"])
    with tab1: uploaded = st.file_uploader("Upload WAV/MP3", type=["wav", "mp3"])
    with tab2: recorded = st.audio_input("Record Live Clip")
    source = uploaded if uploaded else recorded

    if source:
        st.audio(source)
        if st.button("🚀 Analyze Audio", use_container_width=True):
            with st.spinner("Decoding spectral features..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(source.getvalue()); path = tmp.name
                res = analyze_v2(path)
                res["filename"] = getattr(source, 'name', 'Live_Recording.wav')
                st.session_state.current_result = res
                st.session_state.history.insert(0, res)
                st.session_state.page = "Instrument Distribution"
                st.rerun()

elif st.session_state.page == "Instrument Distribution":
    res = st.session_state.current_result
    if not res: st.warning("Please analyze a file first!")
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

elif st.session_state.page == "Audio Analysis":
    res = st.session_state.current_result
    if not res: st.warning("Analyze a file to see spectral breakdown.")
    else:
        st.header("📈 Spectral Visualization")
        st.info("The AI Agent redirected you here to show the technical evidence (Mel-Spectrogram).")
        
        fig, ax = plt.subplots(2, 1, figsize=(10, 7))
        librosa.display.waveshow(res['raw_y'], sr=res['sr'], ax=ax[0], color="#2E86C1")
        ax[0].set_title("Time Domain: Waveform")
        S = librosa.feature.melspectrogram(y=res['raw_y'], sr=res['sr'])
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', ax=ax[1])
        ax[1].set_title("Frequency Domain: Mel-Spectrogram")
        plt.tight_layout()
        st.pyplot(fig)

elif st.session_state.page == "History":
    st.header("📜 Session History")
    if not st.session_state.history: st.info("History is empty.")
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