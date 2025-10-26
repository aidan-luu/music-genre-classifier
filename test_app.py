#!/usr/bin/env python3
"""
Simple test script to verify the app components work
"""
import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    try:
        import streamlit as st
        import librosa
        import numpy as np
        import tensorflow as tf
        import pickle
        import plotly.graph_objects as go
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_model_files():
    """Test if model files exist"""
    required_files = ['genre_classifier_model.h5', 'scaler.pkl', 'genre_classes.npy']
    all_exist = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            all_exist = False
    
    return all_exist

def test_feature_extraction():
    """Test feature extraction with a dummy audio"""
    try:
        import librosa
        import numpy as np
        
        # Create a dummy audio signal
        duration = 1  # 1 second
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Test basic librosa features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        print("✓ Feature extraction works")
        return True
    except Exception as e:
        print(f"✗ Feature extraction error: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Music Genre Classifier App...")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_model_files,
        test_feature_extraction
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("✓ App is ready for deployment!")
        return True
    else:
        print("✗ App needs fixes before deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

