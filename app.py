import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, duration=30)
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_mean = np.mean(chroma, axis=1)
    
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    spectral_centroid_mean = np.mean(spectral_centroid)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    spectral_rolloff_mean = np.mean(spectral_rolloff)
    
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
    zcr_mean = np.mean(zero_crossing_rate)
    
    features = np.concatenate([
        mfccs_mean,
        mfccs_std,
        chroma_mean,
        [spectral_centroid_mean, spectral_rolloff_mean, zcr_mean]
    ])
    
    return features, audio, sample_rate

# Streamlit UI
st.title("🎵 Music Genre Classifier")
st.write("Upload an audio file for feature extraction (ML model loading disabled temporarily)")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file (.wav, .mp3)", type=['wav', 'mp3'])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(uploaded_file)
    
    with st.spinner("Extracting audio features..."):
        # Extract features
        features, audio, sr = extract_features("temp_audio.wav")
        
        # Display features
        st.success("Audio features extracted successfully!")
        st.write(f"Extracted {len(features)} features")
        st.write(f"Audio length: {len(audio)/sr:.2f} seconds")
        st.write(f"Sample rate: {sr} Hz")
        
        # Show feature statistics
        st.subheader("Feature Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean", f"{np.mean(features):.4f}")
        with col2:
            st.metric("Std Dev", f"{np.std(features):.4f}")
        with col3:
            st.metric("Min/Max", f"{np.min(features):.4f}/{np.max(features):.4f}")
        
        # Waveform visualization
        st.subheader("Audio Waveform")
        fig = go.Figure()
        time = np.linspace(0, len(audio)/sr, len(audio))
        fig.add_trace(go.Scatter(x=time, y=audio, mode='lines', name='Waveform'))
        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

st.warning("Note: ML model prediction is temporarily disabled due to deployment compatibility issues.")
st.markdown("---")
st.markdown("Built with Librosa and Streamlit")