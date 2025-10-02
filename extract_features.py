import librosa
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def extract_features(file_path):
    """Extract audio features from a single file"""
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, duration=30)
        
        # Extract features
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
        
        # Combine all features
        features = np.concatenate([
            mfccs_mean,
            mfccs_std,
            chroma_mean,
            [spectral_centroid_mean, spectral_rolloff_mean, zcr_mean]
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