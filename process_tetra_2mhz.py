#!/usr/bin/env python3
"""
Step 2-4: Process TETRA Files (2 MHz)
Classify all 38 TETRA files using center 200-bin (2 MHz) chunks.
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
from extract_2mhz import extract_2mhz

def find_tetra_files():
    """Find all TETRA files in dataset."""
    import subprocess
    cmd = 'find dataset/ -name "*.npy" | grep "_tetra_" | sort'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    files = result.stdout.strip().split('\n')
    return [f for f in files if f]

def process_tetra_files_2mhz():
    """Process all TETRA files with 2 MHz extraction and calculate accuracy."""
    print("=" * 60)
    print("Step 2-4: Process TETRA Files (2 MHz)")
    print("=" * 60)
    
    # Find TETRA files
    tetra_files = find_tetra_files()
    print(f"\nFound {len(tetra_files)} TETRA files")
    
    # Load models
    print("\nLoading models...")
    scaler, encoder, model = load_models()
    print("Models loaded")
    
    # Process each file
    print("\nProcessing TETRA files (2 MHz)...")
    print("-" * 60)
    
    correct = 0
    total = len(tetra_files)
    results = []
    
    for i, filepath in enumerate(tetra_files, 1):
        filename = os.path.basename(filepath)
        
        # Load data
        data = np.load(filepath)
        
        # Extract center 200 bins (2 MHz)
        data = extract_2mhz(data)
        
        # Truncate to 50 time segments
        data = data[:50, :].astype(np.float32)
        
        # Extract features
        features, _ = extract_33_features(data)
        
        # Classify
        predicted_label, predicted_class, predictions = classify(features, scaler, encoder, model)
        
        # Compare with ground truth
        ground_truth = 'tetra'
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
    print(f"TETRA: {correct}/{total} = {accuracy:.1f}%")
    
    return results, correct, total

if __name__ == "__main__":
    results, correct, total = process_tetra_files_2mhz()