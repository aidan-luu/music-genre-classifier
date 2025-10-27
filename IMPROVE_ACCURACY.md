# Improving Accuracy from 70% to 85-95%

## Current State
- **Accuracy:** ~70%
- **Method:** Handcrafted features (41 features) + Dense Neural Network
- **Features:** MFCCs, Chroma, Spectral features, Tempo

## State-of-the-Art Approach

### 🎯 Target: 85-95% Accuracy

## Key Improvements

### 1. **Mel Spectrograms (Instead of Handcrafted Features)**
   - **Why:** CNNs can learn better representations than handcrafted features
   - **What:** Convert audio to 2D time-frequency representations
   - **Benefit:** +10-15% accuracy improvement

### 2. **CNN Architecture (Instead of Dense Network)**
   - **Why:** CNNs excel at finding patterns in images/spectrograms
   - **Architecture:**
     - 4 convolutional blocks with batch normalization
     - Global average pooling
     - Dense layers with dropout
   - **Benefit:** +5-10% accuracy improvement

### 3. **Data Augmentation**
   - **Techniques:**
     - Pitch shifting (-2 to +2 semitones)
     - Time stretching (0.8x to 1.2x)
     - Adding random noise
   - **Why:** Increases effective dataset size, prevents overfitting
   - **Benefit:** +3-5% accuracy improvement

### 4. **Modern Training Techniques**
   - Early stopping (prevent overfitting)
   - Learning rate scheduling (adaptive learning)
   - Batch normalization (stable training)
   - Dropout (regularization)

---

## How to Train the Improved Model

### Step 1: Run the Training Script

```bash
python train_cnn_model.py
```

**What it does:**
1. Loads audio files from `genres_original/`
2. Extracts mel spectrograms (128 mel bins)
3. Builds CNN architecture
4. Trains for up to 100 epochs (early stopping enabled)
5. Saves best model as `genre_classifier_cnn.keras`

**Expected time:** 30-60 minutes (depending on hardware)

### Step 2: Check Results

After training completes, check:
- Console output for final accuracy
- `training_history_cnn.png` for training curves
- `genre_classifier_cnn.keras` (the trained model)

### Step 3: Deploy the Improved Model

Update `app_keras.py` to use the CNN model (see instructions below)

---

## Technical Details

### Mel Spectrogram Parameters
```python
Sample Rate: 22050 Hz
Duration: 30 seconds
N Mels: 128
FFT Size: 2048
Hop Length: 512
```

This creates a **128 x ~1300** spectrogram (time-frequency representation)

### CNN Architecture
```
Conv2D(32) -> BN -> MaxPool -> Dropout(0.25)
Conv2D(64) -> BN -> MaxPool -> Dropout(0.25)
Conv2D(128) -> BN -> MaxPool -> Dropout(0.25)
Conv2D(256) -> BN -> MaxPool -> Dropout(0.25)
GlobalAvgPool
Dense(512) -> BN -> Dropout(0.5)
Dense(256) -> BN -> Dropout(0.5)
Dense(10, softmax)
```

**Total parameters:** ~2-3 million (vs 200K in old model)

### Training Configuration
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Sparse Categorical Crossentropy
- **Batch Size:** 32
- **Early Stopping:** Patience=20 epochs
- **LR Reduction:** Factor=0.5, Patience=10

---

## Expected Results

### Baseline (Current Model)
- **Accuracy:** ~70%
- **Method:** 41 handcrafted features + Dense NN

### Improved CNN Model
- **Target:** 85-90% accuracy
- **Method:** Mel spectrograms + CNN

### State-of-the-Art (If you want to go further)
- **Target:** 90-95% accuracy
- **Additional techniques:**
  - Ensemble of multiple models
  - Transfer learning (use pre-trained audio models)
  - Attention mechanisms
  - Larger dataset (augmentation x10)

---

## Integrating CNN Model into Streamlit App

After training, update `app_keras.py`:

### 1. Update Feature Extraction

Replace the current `extract_features()` function with:

```python
def extract_mel_spectrogram(file_path):
    """Extract mel spectrogram for CNN model"""
    audio, sr = librosa.load(file_path, duration=30, sr=22050)

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())

    return mel_spec_norm[..., np.newaxis], audio, sr  # Add channel dimension
```

### 2. Update Model Loading

```python
st.session_state.model = keras.models.load_model('genre_classifier_cnn.keras')
st.session_state.genre_classes = np.load('genre_classes_cnn.npy', allow_pickle=True)
# No scaler needed for CNN model!
```

### 3. Update Prediction

```python
def predict_genre(mel_spec):
    mel_spec = mel_spec[np.newaxis, ...]  # Add batch dimension
    predictions = st.session_state.model.predict(mel_spec, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    predicted_genre = st.session_state.genre_classes[predicted_idx]
    return predicted_genre, predictions
```

---

## Troubleshooting

### "Out of Memory" Error
- Reduce batch size: `BATCH_SIZE = 16`
- Reduce n_mels: `N_MELS = 64`

### Training Takes Too Long
- Reduce epochs: `EPOCHS = 50`
- Use smaller model (remove one conv block)

### Accuracy Not Improving
- Add more data augmentation
- Try different architectures
- Increase training time
- Check for class imbalance

---

## Alternative: Quick Win Improvements

If you don't want to retrain from scratch, try these quick improvements to the current model:

### 1. Add More Features (5-10% boost)
Add delta features (temporal derivatives):
```python
mfcc_delta = librosa.feature.delta(mfccs)
mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
```

### 2. Ensemble Current Models (3-5% boost)
Train 3-5 models with different random seeds and average predictions

### 3. Better Preprocessing (2-3% boost)
- Normalize audio before feature extraction
- Remove silence from audio files
- Apply pre-emphasis filter

---

## Comparison: Methods vs Accuracy

| Method | Accuracy | Training Time | Deployment Size |
|--------|----------|---------------|-----------------|
| Current (41 features) | ~70% | 5 min | 2.2 MB |
| CNN (Mel Spectrograms) | 85-90% | 30-60 min | 10-15 MB |
| Ensemble CNN | 90-95% | 2-3 hours | 30-50 MB |
| Transfer Learning | 92-97% | 1-2 hours | 50-100 MB |

---

## Next Steps

1. **Run training:** `python train_cnn_model.py`
2. **Check accuracy** in console output
3. **If >80%:** Integrate into app
4. **If <80%:** Add more augmentation, train longer

Ready to start? Run the training script! 🚀
