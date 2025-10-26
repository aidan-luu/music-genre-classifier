#!/usr/bin/env python3
"""
Test script to verify deployment readiness
"""
import os
import sys

def test_files_exist():
    """Check if all required files exist"""
    required_files = [
        'app_deploy.py',
        'genre_classifier_model.h5', 
        'scaler.pkl',
        'genre_classes.npy',
        'requirements.txt',
        '.streamlit/config.toml',
        'Procfile',
        'runtime.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def test_imports():
    """Test critical imports without TensorFlow threading issues"""
    try:
        import streamlit as st
        import librosa
        import numpy as np
        import pickle
        import plotly.graph_objects as go
        print("✅ Core imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_loading():
    """Test model loading with proper TensorFlow configuration"""
    try:
        import tensorflow as tf
        import numpy as np
        import pickle
        
        # Configure TensorFlow for deployment
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.set_visible_devices([], 'GPU')
        
        # Test loading model files
        model = tf.keras.models.load_model('genre_classifier_model.h5')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        genre_classes = np.load('genre_classes.npy', allow_pickle=True)
        
        print("✅ Model loading successful")
        print(f"   - Model input shape: {model.input_shape}")
        print(f"   - Number of genres: {len(genre_classes)}")
        print(f"   - Scaler fitted: {hasattr(scaler, 'mean_')}")
        return True
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def main():
    """Run all deployment tests"""
    print("🧪 Testing Music Genre Classifier Deployment Readiness")
    print("=" * 60)
    
    tests = [
        ("File Check", test_files_exist),
        ("Import Test", test_imports), 
        ("Model Loading", test_model_loading)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"   ⚠️  {test_name} failed")
    
    print("\n" + "=" * 60)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 App is READY for deployment!")
        print("\n🚀 Deployment options:")
        print("   1. Streamlit Cloud: Push to GitHub and connect")
        print("   2. Heroku: Use 'git push heroku main'")
        print("   3. Local: Run 'streamlit run app_deploy.py'")
        return True
    else:
        print("❌ App needs fixes before deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
