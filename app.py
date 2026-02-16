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
# 🤖 CHATBOT LOGIC ENGINE
# ==========================================
def get_bot_response(user_input, last_prediction=None):
    user_input = user_input.lower()
    
    if "backend" in user_input or "pipeline" in user_input:
        return "Our backend pipeline: 1. Audio Upload -> 2. Mono Normalization -> 3. Mel Spectrogram generation -> 4. CNN Resize -> 5. Class Prediction."
    
    elif "mel spectrogram" in user_input or "spectrogram" in user_input:
        return "A Mel Spectrogram converts sound into a frequency vs. time format using a Mel scale that mimics human hearing perception."
    
    elif "model" in user_input or "cnn" in user_input:
        return "The CNN architecture features Conv2D layers for feature extraction, MaxPooling for reduction, and Softmax for final classification."
    
    elif "accuracy" in user_input:
        return "The Instrunet model achieves approximately 85–92% accuracy depending on the specific training dataset configuration."
    
    elif "overfitting" in user_input:
        return "We mitigate overfitting using Dropout layers, Early Stopping during training, and strict train-validation data splits."
    
    elif "prediction" in user_input:
        if last_prediction:
            return f"The last predicted instrument was: {last_prediction}. This was mapped via learned spectral patterns."
        return "Please analyze an audio file in the Studio first to generate a prediction."
    
    else:
        return "I can explain the Backend, CNN Model, Mel Spectrograms, or Accuracy. What would you like to know?"

# ==========================================
# 🎨 ANIMATED CSS UI ENGINE
# ==========================================
def apply_ultra_styles():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
        .stApp { background: #0b0f19; color: #e2e8f0; font-family: 'Inter', sans-serif; }
        
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .stMarkdown, .stButton, .stPlotlyChart { animation: fadeInUp 0.6s ease-out; }

        [data-testid="stSidebar"] { background-color: #0f172a !important; border-right: 1px solid #1e293b; }
        .nav-header { color: #38bdf8; font-size: 28px; font-weight: 900; padding: 30px 0; text-align: center; border-bottom: 2px solid #1e293b; }

        .hero-section {
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 28px; padding: 60px; text-align: center; margin: 40px 0;
            backdrop-filter: blur(25px); box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
        }
        .hero-section h1 { font-size: 64px !important; font-weight: 900 !important; color: #ffffff !important; }

        .metric-card {
            background: rgba(30, 41, 59, 0.6); border-radius: 20px; padding: 30px;
            border: 1px solid #334155; text-align: center; margin-bottom: 20px;
        }

        .stButton>button {
            background: linear-gradient(90deg, #0ea5e9 0%, #6366f1 100%);
            border: none; border-radius: 16px; color: white; height: 4.5em; font-weight: 800;
        }
        .ai-msg { background: #1e293b; border-radius: 18px; padding: 15px; margin: 10px 0; border-left: 5px solid #38bdf8; font-size: 0.9em; }
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
        peaks = librosa.util.peak_pick(onset_env, 7, 7, 7, 7, 0.5, 30)
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
# 🖥️ ROUTING FUNCTIONS
# ==========================================
def render_home():
    st.markdown("<div class='hero-section'><h1>INSTRUNET AI</h1><p>Neural Network Model for Instrumentation Classifier</p></div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='metric-card' style='max-width: 1200px; margin: 0 auto;'>
            <h3>System Architecture</h3>
            <p style='font-size:1.1em; color:#cbd5e1; padding: 10px 40px;'>
                Utilizing deep <b>Convolutional Neural Networks (CNN)</b> for high-resolution <b>Spectral Mapping</b>. 
                The system extracts <b>MFCCs</b> from temporal landmarks to generate real-time distributions.
            </p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("OPEN ANALYSIS STUDIO 🚀", use_container_width=True):
        st.session_state.page = "Upload & Analyze"
        st.rerun()

def render_studio(engine):
    st.title("🎙️ Analysis Studio")
    file = st.file_uploader("Select audio source", type=["wav", "mp3"])
    if file:
        st.audio(file)
        if st.button("EXECUTE NEURAL SCAN"):
            with st.status("Initializing Scan..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(file.getvalue()); p = tmp.name
                res = engine.process_signal(p)
                st.session_state.current = res
                st.session_state.history.append(res)
                st.session_state.page = "Instrument Distribution"
                st.rerun()

def render_distribution():
    res = st.session_state.current
    st.title("📊 Analysis Results")
    st.markdown(f"<div class='hero-section' style='padding:30px;'><h2>{res['result']['label'].upper()}</h2><h4>Confidence: {res['result']['conf']*100:.2f}%</h4></div>", unsafe_allow_html=True)
    df = pd.DataFrame(res['data']['dist'].items(), columns=['Inst', 'Val'])
    st.plotly_chart(px.bar(df, x='Inst', y='Val', color='Val', template="plotly_dark"), use_container_width=True)

# ==========================================
# 🚀 MAIN APPLICATION LOOP
# ==========================================
def main():
    apply_ultra_styles()
    engine = InstrunetCoreV3(MODEL_PATH)
    
    if "page" not in st.session_state: st.session_state.page = "Home"
    if "current" not in st.session_state: st.session_state.current = None
    if "history" not in st.session_state: st.session_state.history = []
    if "chat" not in st.session_state: st.session_state.chat = []

    with st.sidebar:
        st.markdown("<div class='nav-header'>🎼 INSTRUNET AI</div>", unsafe_allow_html=True)
        nav = st.radio("NAVIGATE", ["Home", "Upload & Analyze", "Instrument Distribution", "Audit Logs"])
        if nav != st.session_state.page: st.session_state.page = nav; st.rerun()
        
        st.markdown("<div style='margin-top: 50px;'>", unsafe_allow_html=True)
        st.subheader("🤖 AI Technical Guide")
        
        # Display Chat History
        for c in st.session_state.chat[-4:]:
            role_label = "👤 You" if c["role"] == "user" else "🤖 Bot"
            st.markdown(f"<div class='ai-msg'><b>{role_label}:</b><br>{c['content']}</div>", unsafe_allow_html=True)
        
        # Chat Input logic
        if q := st.chat_input("Ask about the CNN..."):
            last_label = st.session_state.current['result']['label'] if st.session_state.current else None
            response = get_bot_response(q, last_label)
            st.session_state.chat.append({"role": "user", "content": q})
            st.session_state.chat.append({"role": "assistant", "content": response})
            st.rerun()

    if st.session_state.page == "Home": render_home()
    elif st.session_state.page == "Upload & Analyze": render_studio(engine)
    elif st.session_state.page == "Instrument Distribution": render_distribution()

if __name__ == "__main__":
    main()
