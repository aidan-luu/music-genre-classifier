import librosa
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def extract_features(file_path):
    """Extract comprehensive audio features from a single file"""
    try:
        # Load audio file
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
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_centroid_std = np.std(spectral_centroid)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        spectral_rolloff_std = np.std(spectral_rolloff)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        spectral_bandwidth_std = np.std(spectral_bandwidth)
        
        # Zero crossing rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zero_crossing_rate)
        zcr_std = np.std(zero_crossing_rate)
        
        # Tempo and beat
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
        
        # Tonnetz (harmonic network)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sample_rate)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
        
        # Root Mean Square Energy
        rms = librosa.feature.rms(y=audio)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)
        spectral_flatness_mean = np.mean(spectral_flatness)
        spectral_flatness_std = np.std(spectral_flatness)
        
        # Ensure all arrays are 1D
        spectral_centroid_mean = float(spectral_centroid_mean)
        spectral_centroid_std = float(spectral_centroid_std)
        spectral_rolloff_mean = float(spectral_rolloff_mean)
        spectral_rolloff_std = float(spectral_rolloff_std)
        spectral_bandwidth_mean = float(spectral_bandwidth_mean)
        spectral_bandwidth_std = float(spectral_bandwidth_std)
        zcr_mean = float(zcr_mean)
        zcr_std = float(zcr_std)
        rms_mean = float(rms_mean)
        rms_std = float(rms_std)
        spectral_flatness_mean = float(spectral_flatness_mean)
        spectral_flatness_std = float(spectral_flatness_std)
        tempo = float(tempo)
        
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
        
        return features
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def create_dataset(data_path):
    """Create dataset from all audio files"""
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    features_list = []
    labels_list = []
    
    for genre in genres:
        genre_path = os.path.join(data_path, genre)
        print(f"\nProcessing {genre}...")
        
        files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
        
        for filename in tqdm(files):
            file_path = os.path.join(genre_path, filename)
            features = extract_features(file_path)
            
            if features is not None:
                features_list.append(features)
                labels_list.append(genre)
    
    # Create DataFrame
    df = pd.DataFrame(features_list)
    df['genre'] = labels_list
    
    return df

if __name__ == "__main__":
    print("Starting feature extraction...")
    data_path = "genres_original"
    
    df = create_dataset(data_path)
    
    # Save to CSV
    df.to_csv('music_features.csv', index=False)
    print(f"\nDone! Extracted features from {len(df)} songs")
    print(f"Feature shape: {df.shape}")