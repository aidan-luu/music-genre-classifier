#!/usr/bin/env python3
"""
Safe test script that doesn't import TensorFlow at startup
"""
import sys
import os

def test_basic_imports():
    """Test basic imports without TensorFlow"""
    try:
        print("Testing basic imports...")
        
        import streamlit as st
        print("✓ streamlit")
        
        import librosa
        print("✓ librosa")
        
        import numpy as np
        print("✓ numpy")
        
        import pickle
        print("✓ pickle")
        
        import plotly.graph_objects as go
        print("✓ plotly")
        
        import soundfile
        print("✓ soundfile")
        
        import sklearn
        print("✓ scikit-learn")
        
        import pandas as pd
        print("✓ pandas")
        
        print("\n✅ All basic imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False

def test_model_files():
    """Test if model files exist"""
    required_files = ['genre_classifier_model.h5', 'scaler.pkl', 'genre_classes.npy']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All model files present")
        return True

def test_tensorflow_lazy():
    """Test TensorFlow loading only when needed"""
    try:
        print("Testing TensorFlow lazy loading...")
        
        # Don't import TensorFlow at module level
        import tensorflow as tf
        
        # Configure TensorFlow
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.set_visible_devices([], 'GPU')
        
        print("✓ TensorFlow configured successfully")
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Music Genre Classifier (Safe Mode)")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Model Files", test_model_files),
        ("TensorFlow Lazy", test_tensorflow_lazy)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"   ⚠️  {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 App is ready for deployment!")
        return True
    else:
        print("❌ Some issues need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


