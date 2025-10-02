import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import pickle
import plotly.graph_objects as go

# Load model and preprocessing objects
@st.cache_resource
def load_model_and_preprocessing():
    model = tf.keras.models.load_model('genre_classifier_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    genre_classes = np.load('genre_classes.npy', allow_pickle=True)
    return model, scaler, genre_classes

def extract_features(file_path):
    """Extract audio features"""
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
st.write("Upload an audio file and I'll predict its genre!")

# Load model
model, scaler, genre_classes = load_model_and_preprocessing()

# File uploader
uploaded_file = st.file_uploader("Choose an audio file (.wav, .mp3)", type=['wav', 'mp3'])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(uploaded_file)
    
    with st.spinner("Analyzing audio..."):
        # Extract features
        features, audio, sr = extract_features("temp_audio.wav")
        
        # Reshape and scale
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # Predict
        predictions = model.predict(features_scaled)[0]
        predicted_genre_idx = np.argmax(predictions)
        predicted_genre = genre_classes[predicted_genre_idx]
        confidence = predictions[predicted_genre_idx] * 100
        
        # Display results
        st.success(f"**Predicted Genre: {predicted_genre.upper()}**")
        st.write(f"Confidence: {confidence:.1f}%")
        
        # Show all probabilities
        st.subheader("Genre Probabilities")
        fig = go.Figure(data=[
            go.Bar(
                x=genre_classes,
                y=predictions * 100,
                marker_color='lightblue'
            )
        ])
        fig.update_layout(
            xaxis_title="Genre",
            yaxis_title="Probability (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Waveform visualization
        st.subheader("Audio Waveform")
        fig2 = go.Figure()
        time = np.linspace(0, len(audio)/sr, len(audio))
        fig2.add_trace(go.Scatter(x=time, y=audio, mode='lines', name='Waveform'))
        fig2.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("Built with TensorFlow, Librosa, and Streamlit")