#!/usr/bin/env python3
"""
Step 6: Process FM Files
Classify all 41 FM files using full band approach.
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

from feature_extraction import extract_33_features
from inference import load_models, classify

def find_fm_files():
    """Find all FM files in dataset."""
    import subprocess
    cmd = 'find dataset/ -name "*.npy" | grep "_fm_" | sort'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    files = result.stdout.strip().split('\n')
    return [f for f in files if f]

def process_fm_files():
    """Process all FM files and calculate accuracy."""
    print("=" * 60)
    print("Step 6: Process FM Files (Full Band)")
    print("=" * 60)
    
    # Find FM files
    fm_files = find_fm_files()
    print(f"\nFound {len(fm_files)} FM files")
    
    # Load models
    print("\nLoading models...")
    scaler, encoder, model = load_models()
    print("Models loaded")
    
    # Process each file
    print("\nProcessing FM files...")
    print("-" * 60)
    
    correct = 0
    total = len(fm_files)
    results = []
    
    for i, filepath in enumerate(fm_files, 1):
        filename = os.path.basename(filepath)
        
        # Load data
        data = np.load(filepath)
        
        # Truncate to 50 time segments
        data = data[:50, :].astype(np.float32)
        
        # Extract features
        features, _ = extract_33_features(data)
        
        # Classify
        predicted_label, predicted_class, predictions = classify(features, scaler, encoder, model)
        
        # Compare with ground truth
        ground_truth = 'fm'
        match = (predicted_label == ground_truth)
        if match:
            correct += 1
        
        # Store result
        results.append({
            'file': filename,
            'predicted': predicted_label,
            'ground_truth': ground_truth,
            'match': match
        })
        
        # Print result
        status = "✓" if match else "✗"
        print(f"File {i:2d}/{total}: {predicted_label:6s} vs {ground_truth:6s} → {status}")
    
    # Calculate accuracy
    accuracy = correct / total * 100
    
    print("-" * 60)
    print(f"\nResults:")
    print(f"FM: {correct}/{total} = {accuracy:.1f}%")
    
    return results, correct, total

if __name__ == "__main__":
    results, correct, total = process_fm_files()
