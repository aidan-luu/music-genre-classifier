import streamlit as st
import librosa
import numpy as np
import pickle
import plotly.graph_objects as go
import os
import tempfile
import warnings

# Suppress warnings and configure environment
warnings.filterwarnings('ignore')
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global variables for model
_model = None
_scaler = None
_genre_classes = None
_model_loading_attempted = False

def load_model_lazy():
    """Load model only when needed to avoid import-time issues"""
    global _model, _scaler, _genre_classes, _model_loading_attempted
    
    if _model_loading_attempted:
        return _model, _scaler, _genre_classes
    
    _model_loading_attempted = True
    
    try:
        # Import TensorFlow only when needed
        import tensorflow as tf
        
        # Configure TensorFlow to avoid macOS threading issues
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.set_visible_devices([], 'GPU')
        
        # Load model
        _model = tf.keras.models.load_model('genre_classifier_model.h5')
        with open('scaler.pkl', 'rb') as f:
            _scaler = pickle.load(f)
        _genre_classes = np.load('genre_classes.npy', allow_pickle=True)
        
        return _model, _scaler, _genre_classes
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def extract_features(file_path):
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
st.write("Upload an audio file and I'll predict its genre using advanced machine learning!")

# Don't load model at startup - only when user clicks a button
model_loaded = False
model, scaler, genre_classes = None, None, None

# Add a button to load the model
if st.button("🚀 Load Model", type="primary"):
    with st.spinner("Loading model..."):
        try:
            model, scaler, genre_classes = load_model_lazy()
            model_loaded = model is not None and scaler is not None and genre_classes is not None
            
            if model_loaded:
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Failed to load model. This may be due to TensorFlow compatibility issues on macOS.")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.info("💡 **Workaround**: Try deploying to Streamlit Cloud or Heroku where this issue doesn't occur.")

if not model_loaded:
    st.warning("⚠️ Model not loaded. Click the 'Load Model' button above to load the model.")
    st.info("💡 **Note**: On macOS, there may be TensorFlow compatibility issues. The app works perfectly when deployed to cloud platforms.")
    
    # Show feature extraction demo
    st.subheader("🔍 Feature Extraction Demo")
    st.write("Even without the model, we can still extract audio features:")
    
    uploaded_file = st.file_uploader("Choose an audio file (.wav, .mp3)", type=['wav', 'mp3'])
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        temp_filename = f"temp_audio.{file_extension}"
        
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio(uploaded_file)
        
        with st.spinner("Extracting features..."):
            features, audio, sr = extract_features(temp_filename)
            
            if features is not None:
                st.success(f"✅ Successfully extracted {len(features)} audio features!")
                
                # Show feature breakdown
                st.subheader("📊 Feature Breakdown")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**MFCCs (26 features):**")
                    st.write("- 13 mean coefficients")
                    st.write("- 13 standard deviation coefficients")
                    
                    st.write("**Chroma Features (24 features):**")
                    st.write("- 12 mean pitch classes")
                    st.write("- 12 standard deviation pitch classes")
                
                with col2:
                    st.write("**Spectral Features (6 features):**")
                    st.write("- Centroid (mean, std)")
                    st.write("- Rolloff (mean, std)")
                    st.write("- Bandwidth (mean, std)")
                    
                    st.write("**Other Features (21 features):**")
                    st.write("- Tonnetz (6), Spectral contrast (7)")
                    st.write("- ZCR (2), RMS (2), Flatness (2)")
                    st.write("- Tempo (1)")
                
                # Waveform visualization
                st.subheader("🎼 Audio Waveform")
                fig = go.Figure()
                time = np.linspace(0, len(audio)/sr, len(audio))
                fig.add_trace(go.Scatter(x=time, y=audio, mode='lines', name='Waveform', line=dict(color='#4ECDC4')))
                fig.update_layout(
                    xaxis_title="Time (s)",
                    yaxis_title="Amplitude",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Clean up
            try:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            except:
                pass
else:
    # File uploader for when model is loaded
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
                
                if features is not None and model_loaded:
                    # Reshape and scale
                    features = features.reshape(1, -1)
                    features_scaled = scaler.transform(features)
                    
                    # Predict
                    predictions = model.predict(features_scaled, verbose=0)[0]
                    predicted_genre_idx = np.argmax(predictions)
                    predicted_genre = genre_classes[predicted_genre_idx]
                    confidence = predictions[predicted_genre_idx] * 100
                    
                    # Display results
                    st.success(f"**🎯 Predicted Genre: {predicted_genre.upper()}**")
                    st.metric("Confidence", f"{confidence:.1f}%")
                    
                    # Show all probabilities
                    st.subheader("📊 Genre Probabilities")
                    fig = go.Figure(data=[
                        go.Bar(
                            x=genre_classes,
                            y=predictions * 100,
                            marker_color=['#FF6B6B' if i == predicted_genre_idx else '#4ECDC4' for i in range(len(genre_classes))],
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
                    fig2.add_trace(go.Scatter(x=time, y=audio, mode='lines', name='Waveform', line=dict(color='#4ECDC4')))
                    fig2.update_layout(
                        xaxis_title="Time (s)",
                        yaxis_title="Amplitude",
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning("Model not loaded - showing feature extraction only")
                    if features is not None:
                        st.write(f"Extracted {len(features)} audio features")
                    
            except Exception as e:
                st.error(f"Error processing audio: {e}")
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
    st.write("This app uses a deep neural network trained on 999 songs to classify music genres.")
    st.write("**Features used:**")
    st.write("- MFCCs (Mel-frequency cepstral coefficients)")
    st.write("- Chroma features")
    st.write("- Spectral features")
    st.write("- Tempo and rhythm")
    st.write("- Harmonic analysis")
    
    st.header("🎵 Supported Genres")
    if genre_classes is not None:
        for genre in genre_classes:
            st.write(f"• {genre.title()}")
    else:
        st.write("• Blues • Classical • Country • Disco")
        st.write("• Hip-hop • Jazz • Metal • Pop")
        st.write("• Reggae • Rock")
    
    st.header("🚀 Deployment")
    st.write("**Ready for deployment on:**")
    st.write("• Streamlit Cloud")
    st.write("• Heroku")
    st.write("• Google Colab")
    st.write("• Any Linux/Windows server")

st.markdown("---")
st.markdown("Built with TensorFlow, Librosa, and Streamlit | Enhanced with 77 audio features")