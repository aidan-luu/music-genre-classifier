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

def extract_features(file_path):
    """Extract audio features safely"""
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
st.write("Upload an audio file and I'll analyze its features!")

# Check if model files exist
model_files_exist = all([
    os.path.exists('genre_classifier_model.h5'),
    os.path.exists('scaler.pkl'),
    os.path.exists('genre_classes.npy')
])

if not model_files_exist:
    st.warning("⚠️ Model files not found. Running in feature extraction mode only.")
    model_loaded = False
    genre_classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
else:
    st.success("✅ Model files found!")
    model_loaded = True
    try:
        genre_classes = np.load('genre_classes.npy', allow_pickle=True)
    except:
        genre_classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

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
                st.success(f"✅ Successfully extracted {len(features)} audio features!")
                
                # Show feature summary
                st.subheader("📊 Feature Analysis")
                
                # Create feature categories
                feature_categories = {
                    'MFCCs (Mean)': features[:13],
                    'MFCCs (Std)': features[13:26],
                    'Chroma (Mean)': features[26:38],
                    'Chroma (Std)': features[38:50],
                    'Tonnetz': features[50:56],
                    'Spectral Contrast': features[56:63],
                    'Spectral Features': features[63:69],
                    'Energy & Tempo': features[69:77]
                }
                
                # Display feature statistics
                for category, values in feature_categories.items():
                    st.write(f"**{category}**: {len(values)} features, range: [{np.min(values):.3f}, {np.max(values):.3f}]")
                
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
                
                # Feature visualization
                st.subheader("🔍 Feature Distribution")
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(feature_categories.keys()),
                        y=[len(values) for values in feature_categories.values()],
                        marker_color='#4ECDC4'
                    )
                ])
                fig.update_layout(
                    xaxis_title="Feature Category",
                    yaxis_title="Number of Features",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                if not model_loaded:
                    st.info("💡 To enable genre prediction, ensure all model files are present in the same directory.")
                
            else:
                st.error("Failed to extract features from the audio file.")
                
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
    st.write("This app extracts comprehensive audio features for music analysis.")
    st.write("**Features extracted:**")
    st.write("- MFCCs (Mel-frequency cepstral coefficients)")
    st.write("- Chroma features")
    st.write("- Spectral features")
    st.write("- Tempo and rhythm")
    st.write("- Harmonic analysis")
    
    st.header("🎵 Supported Genres")
    for genre in genre_classes:
        st.write(f"• {genre.title()}")
    
    if model_loaded:
        st.success("✅ Model ready for prediction")
    else:
        st.warning("⚠️ Running in analysis mode only")

st.markdown("---")
st.markdown("Built with Librosa and Streamlit | Enhanced with 77 audio features")


