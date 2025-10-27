import streamlit as st
import librosa
import numpy as np
import pickle
import plotly.graph_objects as go
import os
import tempfile
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

# Import Keras
try:
    import keras
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    st.error("❌ Keras not available")

def load_model():
    """Load Keras model into session state"""
    try:
        # Load Keras model
        model_path = 'genre_classifier_model.h5'

        if not os.path.exists(model_path):
            st.error(f"❌ Model not found: {model_path}")
            return False

        st.session_state.model = keras.models.load_model(model_path)

        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            st.session_state.scaler = pickle.load(f)

        # Load genre classes
        st.session_state.genre_classes = np.load('genre_classes.npy', allow_pickle=True)

        st.success("✅ Model loaded successfully!")
        st.info(f"🎵 Ready to classify {len(st.session_state.genre_classes)} genres")

        return True

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False

def predict_genre(features):
    """Make prediction using Keras model"""
    if 'model' not in st.session_state or 'scaler' not in st.session_state:
        return None, None

    try:
        # Reshape and scale features
        features = features.reshape(1, -1)
        features_scaled = st.session_state.scaler.transform(features)

        # Run prediction
        predictions = st.session_state.model.predict(features_scaled, verbose=0)[0]

        # Get predicted genre
        predicted_idx = np.argmax(predictions)
        predicted_genre = st.session_state.genre_classes[predicted_idx]

        return predicted_genre, predictions

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

def extract_features(file_path):
    """Extract audio features from file"""
    try:
        audio, sample_rate = librosa.load(file_path, duration=30)

        # MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        # Chroma features (12 pitch classes)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        spectral_centroid_mean = float(np.mean(spectral_centroid))
        spectral_centroid_std = float(np.std(spectral_centroid))

        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        spectral_rolloff_mean = float(np.mean(spectral_rolloff))
        spectral_rolloff_std = float(np.std(spectral_rolloff))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        spectral_bandwidth_mean = float(np.mean(spectral_bandwidth))
        spectral_bandwidth_std = float(np.std(spectral_bandwidth))

        # Zero crossing rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = float(np.mean(zero_crossing_rate))
        zcr_std = float(np.std(zero_crossing_rate))

        # Tempo and beat
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
        tempo = float(tempo)

        # Tonnetz (harmonic network)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sample_rate)
        tonnetz_mean = np.mean(tonnetz, axis=1)

        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

        # Root Mean Square Energy
        rms = librosa.feature.rms(y=audio)
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))

        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)
        spectral_flatness_mean = float(np.mean(spectral_flatness))
        spectral_flatness_std = float(np.std(spectral_flatness))

        # Combine all features
        features = np.concatenate([
            mfccs_mean,                    # 13 features
            mfccs_std,                     # 13 features
            chroma_mean,                   # 12 features
            chroma_std,                    # 12 features
            tonnetz_mean,                  # 6 features
            spectral_contrast_mean,        # 7 features
            np.array([spectral_centroid_mean, spectral_centroid_std]),      # 2 features
            np.array([spectral_rolloff_mean, spectral_rolloff_std]),        # 2 features
            np.array([spectral_bandwidth_mean, spectral_bandwidth_std]),    # 2 features
            np.array([zcr_mean, zcr_std]),           # 2 features
            np.array([rms_mean, rms_std]),           # 2 features
            np.array([spectral_flatness_mean, spectral_flatness_std]),      # 2 features
            np.array([tempo])                        # 1 feature
        ])

        return features, audio, sample_rate
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None, None, None

# Streamlit UI
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Music Genre Classifier")
st.write("**Powered by Keras - Works on Streamlit Cloud!**")

if not MODEL_AVAILABLE:
    st.error("Keras not installed")
    st.stop()

# Load model
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

if not st.session_state.model_loaded:
    with st.spinner("Loading model..."):
        if load_model():
            st.session_state.model_loaded = True
        else:
            st.error("Failed to load model")
            st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an audio file (.wav, .mp3)", type=['wav', 'mp3'])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    temp_filename = f"temp_audio.{file_extension}"

    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns([1, 2])

    with col1:
        st.audio(uploaded_file)
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")

    with st.spinner("Analyzing audio..."):
        try:
            features, audio, sr = extract_features(temp_filename)

            if features is not None:
                st.success(f"✅ Extracted {len(features)} audio features")

                # Debug info
                st.write(f"Debug: Features shape: {features.shape}")
                st.write(f"Debug: Model loaded: {'model' in st.session_state}")
                st.write(f"Debug: Scaler loaded: {'scaler' in st.session_state}")

                predicted_genre, predictions = predict_genre(features)

                if predicted_genre is not None:
                    predicted_idx = np.argmax(predictions)
                    confidence = predictions[predicted_idx] * 100

                    st.success(f"**🎯 Predicted Genre: {predicted_genre.upper()}**")
                    st.metric("Confidence", f"{confidence:.1f}%")

                    # Genre probabilities
                    st.subheader("📊 Genre Probabilities")
                    fig = go.Figure(data=[
                        go.Bar(
                            x=st.session_state.genre_classes,
                            y=predictions * 100,
                            marker_color=['#FF6B6B' if i == predicted_idx else '#4ECDC4'
                                        for i in range(len(st.session_state.genre_classes))],
                            text=[f"{p*100:.1f}%" for p in predictions],
                            textposition='auto'
                        )
                    ])
                    fig.update_layout(
                        xaxis_title="Genre",
                        yaxis_title="Probability (%)",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Waveform
                    st.subheader("🎼 Audio Waveform")
                    fig2 = go.Figure()
                    time = np.linspace(0, len(audio)/sr, len(audio))
                    fig2.add_trace(go.Scatter(
                        x=time, y=audio, mode='lines',
                        name='Waveform', line=dict(color='#4ECDC4')
                    ))
                    fig2.update_layout(
                        xaxis_title="Time (s)",
                        yaxis_title="Amplitude",
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            try:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            except:
                pass

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("Music genre classification using deep learning")

    st.header("🎵 Genres")
    if 'genre_classes' in st.session_state:
        for genre in st.session_state.genre_classes:
            st.write(f"• {genre.title()}")

st.markdown("---")
st.markdown("Built with Keras, Librosa, and Streamlit")
