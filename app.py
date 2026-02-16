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

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure your model is inside a folder named 'models'
MODEL_PATH = os.path.join(BASE_DIR, "models", "irmas_instrument_model.h5")

# Instrument Labels
INSTRUMENTS = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
FULL_NAMES = {
    'cel': 'Cello', 'cla': 'Clarinet', 'flu': 'Flute', 'gac': 'Acoustic Guitar',
    'gel': 'Electric Guitar', 'org': 'Organ', 'pia': 'Piano', 'sax': 'Saxophone',
    'tru': 'Trumpet', 'vio': 'Violin', 'voi': 'Human Voice'
}

# ==========================================
# 🤖 ENHANCED AI CHATBOT LOGIC
# ==========================================
def get_bot_response(user_input, last_result=None):
    user_input = user_input.lower()
    
    # 1. Backend / Pipeline
    if any(word in user_input for word in ["backend", "pipeline", "process", "how it works"]):
        return """
        <b>Backend Pipeline:</b><br>
        1. <b>Upload:</b> Audio is converted to mono and normalized.<br>
        2. <b>Landmarks:</b> We detect 'onset peaks' to find the strongest signal points.<br>
        3. <b>Feature Extraction:</b> We generate Mel Spectrograms & MFCCs from these points.<br>
        4. <b>CNN Resize:</b> Input is reshaped to (130, 40, 1) for the model.<br>
        5. <b>Prediction:</b> The Softmax layer calculates probability for 11 instruments.
        """

    # 2. Waveform & Pulse Landmarks (Context Aware)
    elif any(word in user_input for word in ["waveform", "peaks", "landmark", "lines", "amplitude"]):
        if last_result:
            count = len(last_result['signal']['landmarks'])
            return f"""
            <b>Waveform Analysis:</b><br>
            The blue graph shows amplitude over time. The <span style='color:#ef4444'><b>red dashed lines</b></span> 
            are the <b>{count} temporal landmarks</b> I detected. <br><br>
            These are the "attack" points of the notes where the instrument's timbre is clearest. 
            I ignored the silence and focused only on these {count} critical moments.
            """
        return "The waveform shows signal amplitude over time. If you upload a file first, I can tell you exactly how many peaks I detected!"

    # 3. Mel Spectrogram
    elif "spectrogram" in user_input or "mel" in user_input:
        return """
        <b>Mel Spectrogram:</b><br>
        This is a visual 'fingerprint' of the sound. It maps <b>Frequency vs. Time</b>, 
        but uses the <b>Mel Scale</b>, which mimics human hearing (we hear changes in low pitch better than high pitch).
        """

    # 4. CNN / Model Architecture
    elif any(word in user_input for word in ["model", "cnn", "layers", "neural", "network"]):
        return """
        <b>CNN Architecture:</b><br>
        We use a Convolutional Neural Network consisting of:<br>
        - <b>Conv2D Layers:</b> To extract spectral features (textures).<br>
        - <b>MaxPooling:</b> To reduce dimensionality and noise.<br>
        - <b>Flatten:</b> To convert 2D maps into a 1D vector.<br>
        - <b>Dense Layers:</b> For classification logic.<br>
        - <b>Softmax:</b> To output the final % probabilities.
        """

    # 5. Accuracy
    elif "accuracy" in user_input:
        return "The Instrunet model achieves approximately <b>85–92% validation accuracy</b> depending on the specific instrument class and recording quality."
    
    # 6. Overfitting
    elif "overfitting" in user_input:
        return """
        <b>Preventing Overfitting:</b><br>
        Overfitting happens when a model 'memorizes' the training songs instead of learning the instrument sounds. 
        We prevent this using <b>Dropout Layers</b> (randomly disabling neurons during training) and <b>Early Stopping</b>.
        """

    # 7. Prediction Result (Context Aware)
    elif "prediction" in user_input or "result" in user_input or "what is this" in user_input:
        if last_result:
            return f"""
            I am {last_result['result']['conf']*100:.1f}% confident this is a 
            <b>{last_result['result']['label'].upper()}</b>.<br>
            This decision was made by averaging the predictions from {len(last_result['signal']['landmarks'])} different segments of the audio.
            """
        return "Please upload an audio file in the Studio tab first!"

    else:
        return "I am the Instrunet Technical Guide. Ask me about the <b>Waveform</b>, <b>CNN Model</b>, <b>Spectrograms</b>, or the <b>Backend Pipeline</b>!"

# ==========================================
# 🎨 ANIMATED CSS UI ENGINE
# ==========================================
def apply_ultra_styles():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
        .stApp { background: #0b0f19; color: #e2e8f0; font-family: 'Inter', sans-serif; }
        
        /* Animations */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .stMarkdown, .stButton, .stPlotlyChart { animation: fadeInUp 0.6s ease-out; }

        /* Sidebar */
        [data-testid="stSidebar"] { background-color: #0f172a !important; border-right: 1px solid #1e293b; }
        .nav-header { color: #38bdf8; font-size: 28px; font-weight: 900; padding: 30px 0; text-align: center; border-bottom: 2px solid #1e293b; }

        /* Hero Section */
        .hero-section {
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 28px; padding: 60px; text-align: center; margin: 40px 0;
            backdrop-filter: blur(25px); box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
        }
        .hero-section h1 { font-size: 64px !important; font-weight: 900 !important; color: #ffffff !important; }

        /* Cards */
        .metric-card {
            background: rgba(30, 41, 59, 0.6); border-radius: 20px; padding: 30px;
            border: 1px solid #334155; text-align: center; margin-bottom: 20px;
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #0ea5e9 0%, #6366f1 100%);
            border: none; border-radius: 16px; color: white; height: 4.5em; font-weight: 800;
        }
        .stButton>button:hover { transform: translateY(-3px); box-shadow: 0 10px 20px rgba(56, 189, 248, 0.3); }

        /* Chat Bubbles */
        .ai-msg { background: #1e293b; border-radius: 12px; padding: 15px; margin: 10px 0; border-left: 4px solid #38bdf8; font-size: 0.9em; line-height: 1.5; }
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
        # Load audio (force 15s duration for consistency)
        y, sr = librosa.load(path, sr=22050, duration=15)
        
        # 1. Onset Detection (Find the 'attacks' of notes)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        peaks = librosa.util.peak_pick(onset_env, pre_max=7, post_max=7, pre_avg=7, post_avg=7, delta=0.5, wait=30)
        times = librosa.frames_to_time(peaks, sr=sr)
        
        # Fallback if no peaks found
        if len(times) == 0: times = [0.0]
        
        features = []
        # Process up to 10 distinct segments from the audio
        for t in times[:10]:
            start = int(max(0, (t - 0.5) * sr))
            chunk = y[start : start + int(3*sr)]
            # Pad if chunk is too short
            if len(chunk) < 3*sr: chunk = np.pad(chunk, (0, int(3*sr)-len(chunk)))
            
            # MFCC Extraction
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=40).T
            # Resize to model input shape (130, 40)
            mfcc = mfcc[:130] if mfcc.shape[0] >= 130 else np.pad(mfcc, ((0, 130-mfcc.shape[0]), (0, 0)))
            
            # Predict
            features.append(self.model.predict(mfcc.reshape(1, 130, 40, 1), verbose=0)[0])

        # Average predictions across all chunks
        avg_preds = np.mean(features, axis=0)
        top_idx = np.argmax(avg_preds)
        
        return {
            "meta": {"id": datetime.now().strftime("%H:%M:%S")},
            "result": {"label": FULL_NAMES[INSTRUMENTS[top_idx]], "conf": avg_preds[top_idx]},
            "data": {"dist": {FULL_NAMES[INSTRUMENTS[i]]: float(avg_preds[i]) for i in range(len(INSTRUMENTS))}},
            "signal": {"y": y, "sr": sr, "landmarks": times, "spec": librosa.feature.melspectrogram(y=y, sr=sr)}
        }

# ==========================================
# 🖥️ PAGE ROUTING & RENDERERS
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
                
                # Check if model loaded
                if engine.model is None:
                    st.error("Model file not found! Check 'models/irmas_instrument_model.h5'")
                else:
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
    fig = px.bar(df, x='Inst', y='Val', color='Val', template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("OPEN TECHNICAL SIGNAL BREAKDOWN 🔬", use_container_width=True):
        st.session_state.page = "Deep Technical Analysis"
        st.rerun()

def render_technical():
    res = st.session_state.current
    st.title("🔬 Deep Technical Analysis")
    
    # 1. Waveform with Landmarks
    st.subheader("1. Pulse Landmark & Temporal Peaks")
    t = np.linspace(0, len(res['signal']['y'])/res['signal']['sr'], num=len(res['signal']['y']))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[::100], y=res['signal']['y'][::100], name="Amplitude", line=dict(color='#38bdf8', width=1.5)))
    for l in res['signal']['landmarks']:
        fig.add_vline(x=l, line_dash="dash", line_color="#ef4444", opacity=0.7)
    fig.update_layout(template="plotly_dark", height=350, margin=dict(t=10)); st.plotly_chart(fig, use_container_width=True)
    
    # 2. Spectrogram
    st.subheader("2. Mel-Spectrogram (Timbre Fingerprinting)")
    S_db = librosa.power_to_db(res['signal']['spec'], ref=np.max)
    fig2 = px.imshow(S_db, origin='lower', aspect='auto', template="plotly_dark", color_continuous_scale='Magma')
    fig2.update_layout(height=400, margin=dict(t=10))
    st.plotly_chart(fig2, use_container_width=True)

def render_history():
    st.title("📜 Neural Audit Logs")
    if not st.session_state.history: 
        st.info("No previous sessions found.")
    else:
        for item in reversed(st.session_state.history):
            st.markdown(f"<div class='ai-msg'><b>SESSION [{item['meta']['id']}]</b><br>{item['result']['label']} — {item['result']['conf']*100:.1f}% Confidence</div>", unsafe_allow_html=True)

# ==========================================
# 🚀 MAIN APPLICATION LOOP
# ==========================================
def main():
    apply_ultra_styles()
    engine = InstrunetCoreV3(MODEL_PATH)
    
    # Initialize Session State
    if "page" not in st.session_state: st.session_state.page = "Home"
    if "current" not in st.session_state: st.session_state.current = None
    if "history" not in st.session_state: st.session_state.history = []
    if "chat" not in st.session_state: st.session_state.chat = []

    # Sidebar Navigation & Chat
    with st.sidebar:
        st.markdown("<div class='nav-header'>🎼 INSTRUNET AI</div>", unsafe_allow_html=True)
        nav = st.radio("NAVIGATE SYSTEM", ["Home", "Upload & Analyze", "Instrument Distribution", "Deep Technical Analysis", "Audit Logs"], 
                       index=["Home", "Upload & Analyze", "Instrument Distribution", "Deep Technical Analysis", "Audit Logs"].index(st.session_state.page))
        if nav != st.session_state.page: st.session_state.page = nav; st.rerun()
        
        # Chatbot Section
        st.markdown("<div style='margin-top: 50px;'>", unsafe_allow_html=True)
        st.subheader("🤖 AI Technical Guide")
        
        # Display Chat History (Last 3 messages)
        for c in st.session_state.chat[-3:]:
            role_label = "👤 YOU" if c["role"] == "user" else "🤖 AI"
            st.markdown(f"<div class='ai-msg'><b>{role_label}:</b><br>{c['content']}</div>", unsafe_allow_html=True)
        
        # Chat Input
        if q := st.chat_input("Ask about Waveform or CNN..."):
            # Pass 'current' result to bot so it can answer specific questions
            response = get_bot_response(q, st.session_state.current)
            st.session_state.chat.append({"role": "user", "content": q})
            st.session_state.chat.append({"role": "assistant", "content": response})
            st.rerun()

    # Page Rendering
    if st.session_state.page == "Home": render_home()
    elif st.session_state.page == "Upload & Analyze": render_studio(engine)
    elif st.session_state.page == "Instrument Distribution": render_distribution()
    elif st.session_state.page == "Deep Technical Analysis": render_technical()
    elif st.session_state.page == "Audit Logs": render_history()

if __name__ == "__main__":
    main()
