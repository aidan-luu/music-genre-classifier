# Music Genre Classifier

A deep learning web application that predicts music genres from audio files using neural networks and audio feature extraction.

## Demo
[Live Demo]([https://music-genre-classifier-aluu.streamlit.app/](https://huggingface.co/spaces/nontwan/music-genre-classifier))

## Features
- Classifies music into 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- Trained on 1,000 songs using deep neural networks
- Achieves 69% accuracy on test set
- Real-time audio analysis with interactive visualizations
- Waveform display and confidence scores

## Technologies
- **Python** - Core programming language
- **TensorFlow** - Deep learning framework
- **Librosa** - Audio feature extraction (MFCCs, spectral features, chroma)
- **Streamlit** - Web application framework
- **Plotly** - Interactive visualizations
- **Scikit-learn** - Data preprocessing and splitting

## Model Architecture
- Deep neural network with 4 layers (512 → 256 → 128 → 10)
- Dropout regularization (0.3) to prevent overfitting
- Adam optimizer with sparse categorical crossentropy loss
- Trained for 100 epochs on GTZAN dataset

## Features Extracted
- MFCCs (Mel-frequency cepstral coefficients) - 13 coefficients with mean and std
- Chroma features - 12 pitch classes
- Spectral centroid - brightness of sound
- Spectral rolloff - frequency below which 85% of energy is contained
- Zero crossing rate - percussiveness indicator

## Local Setup
```bash
# Clone repository
git clone https://github.com/yourusername/music-genre-classifier.git
cd music-genre-classifier

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
