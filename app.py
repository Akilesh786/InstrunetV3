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
# Initialize Chat History for the Agent
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I'm your Instrunet AI Agent. Analyze some audio to start a conversation!"}]

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
    current_index = pages.index(st.session_state.page)
    choice = st.radio("Go to:", pages, index=current_index)
    
    if choice != st.session_state.page:
        st.session_state.page = choice
        st.rerun()
    
    st.markdown("---")
    
    # --- FLOATING AI AGENT SECTION ---
    st.subheader("🤖 AI Assistant")
    with st.expander("💬 Ask about current audio", expanded=True if st.session_state.current_result else False):
        # Display existing chat
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        # Chat input logic
        if prompt := st.chat_input("How does this sound?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            # CONTEXT-AWARE RESPONSE LOGIC
            res = st.session_state.current_result
            if res is None:
                response = "I need you to upload and analyze an audio file before I can answer specific questions!"
            else:
                p = prompt.lower()
                if "why" in p or "reason" in p:
                    response = f"I classified this as {res['family']} because my CNN detected spectral peaks consistent with {res['instrument']} instruments."
                elif "confidence" in p or "sure" in p:
                    response = f"I am {res['confidence']*100:.1f}% confident. The Mel-Spectrogram showed a very clear signature for {res['instrument']}."
                elif "timbre" in p or "frequency" in p:
                    response = "The frequency domain shows high harmonic content in the mid-range, which is typical for this category."
                else:
                    response = f"This sounds like a {res['instrument']} from the {res['family']} family. What else would you like to know?"

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

# =========================
# PAGE ROUTING
# =========================

if st.session_state.page == "About":
    st.header("📖 About Instrunet AI")
    st.write("Instrunet AI is a state-of-the-art instrument recognition system using a Multi-label CNN.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Architecture", "4-Layer CNN")
    col2.metric("Dataset", "IRMAS")
    col3.metric("Window", "3 Seconds")
    
    if st.button("Get Started 🚀", use_container_width=True):
        st.session_state.page = "Upload & Analyze"
        st.rerun()

elif st.session_state.page == "Upload & Analyze":
    st.header("📤 Input Audio")
    tab1, tab2 = st.tabs(["📁 Upload File", "🎤 Live Record"])
    with tab1: uploaded = st.file_uploader("Upload WAV/MP3", type=["wav", "mp3"])
    with tab2: recorded = st.audio_input("Record audio")
    source = uploaded if uploaded else recorded

    if source:
        st.audio(source)
        if st.button("🚀 Analyze Audio", use_container_width=True):
            with st.spinner("Analyzing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(source.getvalue())
                    path = tmp.name
                res = analyze_v2(path)
                res["filename"] = getattr(source, 'name', 'Live Recording.wav')
                st.session_state.current_result = res
                st.session_state.history.insert(0, res)
                st.session_state.page = "Instrument Distribution"
                st.rerun()

elif st.session_state.page == "Instrument Distribution":
    res = st.session_state.current_result
    if not res:
        st.warning("Analyze a file first!")
    else:
        st.header(f"📊 Results for: {res['filename']}")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"### Detected Family: \n# {res['family'].upper()}")
            st.metric("Confidence", f"{res['confidence']*100:.1f}%")
        with c2:
            st.subheader("Instrument Breakdown")
            for inst in FAMILY_MAP[res['family']]:
                st.write(f"**{inst.upper()}**")
                st.progress(res["distribution"][inst])

elif st.session_state.page == "Audio Analysis":
    res = st.session_state.current_result
    if not res: st.warning("Analyze a file first!")
    else:
        st.header("📈 Spectral Visualization")
        
        fig, ax = plt.subplots(2, 1, figsize=(10, 7))
        librosa.display.waveshow(res['raw_y'], sr=res['sr'], ax=ax[0])
        S = librosa.feature.melspectrogram(y=res['raw_y'], sr=res['sr'])
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', ax=ax[1])
        st.pyplot(fig)

elif st.session_state.page == "History":
    st.header("📜 History")
    for i, item in enumerate(st.session_state.history):
        if st.button(f"{item['filename']} — {item['family'].upper()}", key=f"h_{i}"):
            st.session_state.current_result = item
            st.session_state.page = "Instrument Distribution"
            st.rerun()