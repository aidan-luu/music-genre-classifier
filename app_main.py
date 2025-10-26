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

# Import TFLite - try multiple approaches
TFLITE_AVAILABLE = False
tflite = None

try:
    # Try option 1: TensorFlow
    import tensorflow as tf
    tflite = tf.lite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        # Try option 2: Keras 3.0+ has built-in TFLite support
        import keras
        # For keras, we'll use a different approach
        TFLITE_AVAILABLE = True
        USE_KERAS = True
    except ImportError:
        st.error("❌ Neither TensorFlow nor Keras available. Cannot load model.")

# Global variables for model
_interpreter = None
_scaler = None
_genre_classes = None
_input_details = None
_output_details = None

def load_tflite_model():
    """Load TensorFlow Lite model (no mutex lock issues!)"""
    global _interpreter, _scaler, _genre_classes, _input_details, _output_details

    try:
        # Check if TFLite model exists
        tflite_path = 'genre_classifier_model.tflite'

        if not os.path.exists(tflite_path):
            st.error(f"❌ TFLite model not found: {tflite_path}")
            st.info("💡 Please run convert_to_tflite.py on a platform where TensorFlow works (Linux/Cloud)")
            return False

        # Load TFLite interpreter
        if tflite:
            # Using TensorFlow
            _interpreter = tflite.Interpreter(model_path=tflite_path)
            _interpreter.allocate_tensors()
        else:
            # Using Keras - load as h5 instead since Keras doesn't have TFLite interpreter
            import keras
            # Actually, let's just convert on the fly
            st.warning("Using Keras - loading .h5 model instead of .tflite")
            model_h5_path = 'genre_classifier_model.h5'
            if os.path.exists(model_h5_path):
                _interpreter = keras.models.load_model(model_h5_path)
            else:
                st.error("Neither .tflite nor .h5 model found")
                return False

        # Get input and output details
        _input_details = _interpreter.get_input_details()
        _output_details = _interpreter.get_output_details()

        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            _scaler = pickle.load(f)

        # Load genre classes
        _genre_classes = np.load('genre_classes.npy', allow_pickle=True)

        st.success("✅ TensorFlow Lite model loaded successfully!")
        st.info(f"🎵 Ready to classify {len(_genre_classes)} genres")

        return True

    except Exception as e:
        st.error(f"❌ Error loading TFLite model: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False

def predict_genre(features):
    """Make prediction using TFLite model"""
    global _interpreter, _scaler, _input_details, _output_details

    if _interpreter is None or _scaler is None:
        return None, None

    try:
        # Reshape and scale features
        features = features.reshape(1, -1)
        features_scaled = _scaler.transform(features)

        # Convert to float32 (TFLite requirement)
        features_scaled = features_scaled.astype(np.float32)

        # Set input tensor
        _interpreter.set_tensor(_input_details[0]['index'], features_scaled)

        # Run inference
        _interpreter.invoke()

        # Get output tensor
        predictions = _interpreter.get_tensor(_output_details[0]['index'])[0]

        # Get predicted genre
        predicted_idx = np.argmax(predictions)
        predicted_genre = _genre_classes[predicted_idx]
        confidence = predictions[predicted_idx] * 100

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
st.write("**Powered by TensorFlow Lite - No macOS mutex lock issues!**")

# Check for TFLite model
if not os.path.exists('genre_classifier_model.tflite'):
    st.warning("⚠️ TFLite model not found!")
    st.info("""
    **To convert your model:**

    1. Run the conversion on a Linux system or cloud platform:
       ```bash
       python convert_to_tflite.py
       ```

    2. This will create `genre_classifier_model.tflite`

    3. Copy it back to this directory

    **Alternative:** Use the Docker conversion method (instructions in README)
    """)
    st.stop()

# Load model
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

if not st.session_state.model_loaded:
    with st.spinner("Loading TensorFlow Lite model..."):
        if load_tflite_model():
            st.session_state.model_loaded = True
        else:
            st.error("Failed to load model. Please check the error messages above.")
            st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an audio file (.wav, .mp3)", type=['wav', 'mp3'])

if uploaded_file is not None:
    # Determine file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    temp_filename = f"temp_audio.{file_extension}"

    # Save uploaded file temporarily
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns([1, 2])

    with col1:
        st.audio(uploaded_file)
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")

    with st.spinner("Analyzing audio..."):
        try:
            # Extract features
            features, audio, sr = extract_features(temp_filename)

            if features is not None:
                st.success(f"✅ Extracted {len(features)} audio features")

                # Make prediction
                predicted_genre, predictions = predict_genre(features)

                if predicted_genre is not None:
                    # Display results
                    predicted_idx = np.argmax(predictions)
                    confidence = predictions[predicted_idx] * 100

                    st.success(f"**🎯 Predicted Genre: {predicted_genre.upper()}**")
                    st.metric("Confidence", f"{confidence:.1f}%")

                    # Show all probabilities
                    st.subheader("📊 Genre Probabilities")
                    fig = go.Figure(data=[
                        go.Bar(
                            x=_genre_classes,
                            y=predictions * 100,
                            marker_color=['#FF6B6B' if i == predicted_idx else '#4ECDC4'
                                        for i in range(len(_genre_classes))],
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

                    # Waveform visualization
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
            st.error(f"Error processing audio: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            # Clean up temp file
            try:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            except:
                pass

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app uses TensorFlow Lite for music genre classification.")
    st.write("**Why TensorFlow Lite?**")
    st.write("✓ Works perfectly on macOS")
    st.write("✓ No mutex lock issues")
    st.write("✓ Faster inference")
    st.write("✓ Smaller model size")

    st.header("🎵 Supported Genres")
    if _genre_classes is not None:
        for genre in _genre_classes:
            st.write(f"• {genre.title()}")

    st.header("🚀 Deployment")
    st.write("**Compatible with:**")
    st.write("• macOS (Apple Silicon & Intel)")
    st.write("• Linux")
    st.write("• Windows")
    st.write("• Streamlit Cloud")
    st.write("• Docker containers")

st.markdown("---")
st.markdown("Built with TensorFlow Lite, Librosa, and Streamlit | 77 audio features")
