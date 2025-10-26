#!/usr/bin/env python3
"""
Test all dependencies for the Music Genre Classifier
"""
import sys

def test_imports():
    """Test all required imports"""
    try:
        print("Testing imports...")
        
        import streamlit as st
        print("✓ streamlit")
        
        import librosa
        print("✓ librosa")
        
        import numpy as np
        print("✓ numpy")
        
        import tensorflow as tf
        print("✓ tensorflow")
        
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
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False

def test_tensorflow_config():
    """Test TensorFlow configuration"""
    try:
        import tensorflow as tf
        
        # Configure TensorFlow
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.set_visible_devices([], 'GPU')
        
        print("✓ TensorFlow configured successfully")
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow configuration error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Music Genre Classifier Dependencies")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("TensorFlow Config", test_tensorflow_config)
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
        print("🎉 All dependencies are working!")
        return True
    else:
        print("❌ Some dependencies need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


