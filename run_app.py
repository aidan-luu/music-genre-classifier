#!/usr/bin/env python3
"""
Startup script for the Music Genre Classifier
This script handles TensorFlow configuration before starting Streamlit
"""
import os
import sys
import subprocess

def configure_environment():
    """Configure environment variables for deployment"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    
    # Configure TensorFlow threading
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

def main():
    """Main function to start the app"""
    print("🎵 Starting Music Genre Classifier...")
    
    # Configure environment
    configure_environment()
    
    # Check if model files exist
    required_files = ['genre_classifier_model.h5', 'scaler.pkl', 'genre_classes.npy']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing model files: {missing_files}")
        print("Please ensure all model files are present before running the app.")
        sys.exit(1)
    
    print("✅ All model files found")
    print("🚀 Starting Streamlit app...")
    
    # Start Streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app_final.py',
            '--server.headless', 'true',
            '--server.port', '8501',
            '--server.address', '0.0.0.0'
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
