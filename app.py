import streamlit as st
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
import time
from datetime import datetime

# ==========================================
# 🚩 SYSTEM CORE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Instrunet AI",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "irmas_instrument_model.h5")
INSTRUMENTS = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']

FULL_NAMES = {
    'cel': 'Cello', 'cla': 'Clarinet', 'flu': 'Flute', 'gac': 'Acoustic Guitar',
    'gel': 'Electric Guitar', 'org': 'Organ', 'pia': 'Piano', 'sax': 'Saxophone',
    'tru': 'Trumpet', 'vio': 'Violin', 'voi': 'Human Voice'
}

# ==========================================
# 🎨 ANIMATED CSS UI ENGINE
# ==========================================
def apply_ultra_styles():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');

        .stApp { background: #0b0f19; color: #e2e8f0; font-family: 'Inter', sans-serif; }
        
        /* Smooth Entry Animation */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .stMarkdown, .stButton, .stPlotlyChart {
            animation: fadeInUp 0.6s ease-out;
        }

        [data-testid="stSidebar"] { 
            background-color: #0f172a !important; 
            border-right: 1px solid #1e293b;
            min-width: 300px !important;
        }
        
        .nav-header { 
            color: #38bdf8; font-size: 28px; font-weight: 900; 
            padding: 30px 0; text-align: center; border-bottom: 2px solid #1e293b;
            margin-bottom: 20px;
        }

        /* Hero Container with Spacing */
        .hero-section {
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 28px;
            padding: 60px;
            text-align: center;
            margin: 40px 0;
            backdrop-filter: blur(25px);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
        }
        .hero-section h1 { 
            font-size: 64px !important; font-weight: 900 !important; 
            margin-bottom: 15px !important; color: #ffffff !important;
            letter-spacing: -2px;
        }

        .metric-card {
            background: rgba(30, 41, 59, 0.6); border-radius: 20px; padding: 30px;
            border: 1px solid #334155; text-align: center;
            margin-bottom: 20px;
            transition: 0.3s;
        }
        .metric-card:hover { border-color: #38bdf8; background: rgba(56, 189, 248, 0.05); }

        /* Enhanced Animated Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #0ea5e9 0%, #6366f1 100%);
            border: none; border-radius: 16px; color: white;
            height: 4.5em; font-weight: 800; font-size: 1.1em;
            margin: 20px 0; transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        .stButton>button:hover {
            transform: translateY(-5px) scale(1.02);
            box-shadow: 0 15px 30px rgba(56, 189, 248, 0.4);
            filter: brightness(1.1);
        }

        /* Large Sidebar Radio Options */
        div[data-testid="stSidebarUserContent"] label {
            font-size: 1.2em !important; font-weight: 700 !important;
            padding: 10px 0 !important;
        }

        .ai-msg { background: #1e293b; border-radius: 18px; padding: 20px; margin: 15px 0; border-left: 6px solid #38bdf8; line-height: 1.6; }
        
        hr { border: 0; height: 1px; background: #1e293b; margin: 40px 0; }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 🧠 AI ANALYTICS ENGINE
# ==========================================
class InstrunetCoreV3:
    def __init__(self, path):
        self.model = self._load_model(path)

    @st.cache_resource
    def _load_model(_self, path):
        if os.path.exists(path):
            return tf.keras.models.load_model(path, compile=False)
        return None

    def process_signal(self, path):
        y, sr = librosa.load(path, sr=22050, duration=15)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        peaks = librosa.util.peak_pick(
            onset_env, pre_max=7, post_max=7, pre_avg=7, post_avg=7, delta=0.5, wait=30
        )
        times = librosa.frames_to_time(peaks, sr=sr)
        if len(times) == 0: times = [0.0]
        
        features = []
        for t in times[:10]:
            start = int(max(0, (t - 0.5) * sr))
            chunk = y[start : start + int(3*sr)]
            if len(chunk) < 3*sr: chunk = np.pad(chunk, (0, int(3*sr)-len(chunk)))
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=40).T
            mfcc = mfcc[:130] if mfcc.shape[0] >= 130 else np.pad(mfcc, ((0, 130-mfcc.shape[0]), (0, 0)))
            features.append(self.model.predict(mfcc.reshape(1, 130, 40, 1), verbose=0)[0])

        avg_preds = np.mean(features, axis=0)
        top_idx = np.argmax(avg_preds)
        
        return {
            "meta": {"id": datetime.now().strftime("%H:%M:%S")},
            "result": {"label": FULL_NAMES[INSTRUMENTS[top_idx]], "conf": avg_preds[top_idx]},
            "data": {"dist": {FULL_NAMES[INSTRUMENTS[i]]: float(avg_preds[i]) for i in range(len(INSTRUMENTS))}},
            "signal": {"y": y, "sr": sr, "landmarks": times, "spec": librosa.feature.melspectrogram(y=y, sr=sr)}
        }

# ==========================================
# 🖥️ ROUTING
# ==========================================
def render_home():
    st.markdown("<div class='hero-section'><h1>INSTRUNET AI</h1><p style='font-size:1.2em; opacity:0.8;'>Neural Network Model for Instrumentation Classifier</p></div>", unsafe_allow_html=True)
    
    # Consolidated Architecture Box
    st.markdown("""
        <div class='metric-card' style='max-width: 1200px; margin: 0 auto;'>
            <h3>System Architecture</h3>
            <p style='font-size:1.15em; line-height:1.7; color:#cbd5e1; padding: 15px 40px;'>
                The <b>Instrunet AI</b> framework utilizes a deep <b>Convolutional Neural Network (CNN)</b> architecture specifically optimized for 
                high-resolution <b>Spectral Peak Mapping</b>. By extracting <b>MFCC (Mel-Frequency Cepstral Coefficients)</b> from temporal landmarks, 
                the real-time inference engine generates precise probability distributions for multi-instrument classification.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin: 40px 0;'>", unsafe_allow_html=True)
    if st.button("OPEN ANALYSIS STUDIO 🚀", use_container_width=True):
        st.session_state.page = "Upload & Analyze"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def render_studio(engine):
    st.title("🎙️ Analysis Studio")
    st.markdown("<br>", unsafe_allow_html=True)
    t1, t2 = st.tabs(["📁 UPLOAD MASTER FILE", "🎤 RECORD LIVE SESSION"])
    with t1: file = st.file_uploader("Select high-fidelity audio source", type=["wav", "mp3"])
    with t2: rec = st.audio_input("Initiate stream capture")
    
    src = file if file else rec
    if src:
        st.markdown("<div class='ai-msg'>Signal received. Preview the waveform below before processing.</div>", unsafe_allow_html=True)
        st.audio(src)
        if st.button("EXECUTE NEURAL SCAN"):
            with st.status("Initializing Neural Layers...", expanded=True) as s:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(src.getvalue()); p = tmp.name
                res = engine.process_signal(p)
                st.session_state.current = res
                st.session_state.history.append(res)
                s.update(label="Scanning Complete!", state="complete")
                time.sleep(0.5)
                st.session_state.page = "Instrument Distribution"
                st.rerun()

def render_distribution():
    res = st.session_state.current
    if not res: st.warning("Please analyze a file first."); return
    st.title("📊 Analysis Results")
    st.markdown(f"<div class='hero-section' style='padding:30px;'><h2>{res['result']['label'].upper()}</h2>"
                f"<h4>Neural Confidence: {res['result']['conf']*100:.2f}%</h4></div>", unsafe_allow_html=True)
    
    df = pd.DataFrame(res['data']['dist'].items(), columns=['Inst', 'Val'])
    fig = px.bar(df, x='Inst', y='Val', color='Val', color_continuous_scale='Turbo', template="plotly_dark")
    fig.update_layout(height=450, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("OPEN TECHNICAL SIGNAL BREAKDOWN 🔬", use_container_width=True):
        st.session_state.page = "Deep Technical Analysis"
        st.rerun()

def render_technical():
    res = st.session_state.current
    if not res: st.error("No active session found."); return
    st.title("🔬 Deep Technical Analysis")
    
    st.subheader("1. Pulse Landmark & Temporal Peaks")
    t = np.linspace(0, len(res['signal']['y'])/res['signal']['sr'], num=len(res['signal']['y']))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[::100], y=res['signal']['y'][::100], name="Amplitude", line=dict(color='#38bdf8', width=1.5)))
    for l in res['signal']['landmarks']:
        fig.add_vline(x=l, line_dash="dash", line_color="#ef4444", opacity=0.7)
    fig.update_layout(template="plotly_dark", height=350, margin=dict(t=10)); st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.subheader("2. Mel-Spectrogram (Timbre Fingerprinting)")
    S_db = librosa.power_to_db(res['signal']['spec'], ref=np.max)
    fig2 = px.imshow(S_db, origin='lower', aspect='auto', template="plotly_dark", color_continuous_scale='Magma')
    fig2.update_layout(height=400, margin=dict(t=10))
    st.plotly_chart(fig2, use_container_width=True)

def render_history():
    st.title("📜 Neural Audit Logs")
    if not st.session_state.history: st.info("No previous sessions found in this instance.")
    else:
        for item in reversed(st.session_state.history):
            st.markdown(f"<div class='ai-msg'><b>SESSION [{item['meta']['id']}]</b><br>{item['result']['label']} — {item['result']['conf']*100:.1f}% Confidence</div>", unsafe_allow_html=True)

# ==========================================
# 🚀 MAIN LOOP
# ==========================================
def main():
    apply_ultra_styles()
    engine = InstrunetCoreV3(MODEL_PATH)
    
    if "page" not in st.session_state: st.session_state.page = "Home"
    if "current" not in st.session_state: st.session_state.current = None
    if "history" not in st.session_state: st.session_state.history = []
    if "chat" not in st.session_state: st.session_state.chat = []

    with st.sidebar:
        # Title updated to Instrunet AI
        st.markdown("<div class='nav-header'>🎼 INSTRUNET AI</div>", unsafe_allow_html=True)
        nav = st.radio("NAVIGATE SYSTEM", ["Home", "Upload & Analyze", "Instrument Distribution", "Deep Technical Analysis", "Audit Logs"], 
                       index=["Home", "Upload & Analyze", "Instrument Distribution", "Deep Technical Analysis", "Audit Logs"].index(st.session_state.page))
        if nav != st.session_state.page: st.session_state.page = nav; st.rerun()
        
        st.markdown("<div style='margin-top: 60px;'>", unsafe_allow_html=True)
        st.subheader("🤖 AI Technical Guide")
        for c in st.session_state.chat[-2:]: 
            st.markdown(f"<div class='ai-msg' style='font-size:0.85em; padding:12px;'>{c['content']}</div>", unsafe_allow_html=True)
        
        if q := st.chat_input("Ask about MFCCs..."):
            st.session_state.chat.append({"role": "user", "content": q})
            st.session_state.chat.append({"role": "assistant", "content": "The system identifies timbre by mapping harmonic overtones onto a 2D Mel-frequency grid."})
            st.rerun()

    # Page Routing
    if st.session_state.page == "Home": render_home()
    elif st.session_state.page == "Upload & Analyze": render_studio(engine)
    elif st.session_state.page == "Instrument Distribution": render_distribution()
    elif st.session_state.page == "Deep Technical Analysis": render_technical()
    elif st.session_state.page == "Audit Logs": render_history()

if __name__ == "__main__":
    main()
