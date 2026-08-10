#!/usr/bin/env python3
"""
Step 8: Analyze Misclassifications
Detailed analysis of prediction errors.
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

from feature_extraction import extract_33_features
from inference import load_models, classify, TECH_LABELS

def find_files(technology):
    """Find all .npy files for a given technology."""
    import subprocess
    cmd = f'find dataset/ -name "*.npy" | grep "_{technology}_" | sort'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    files = result.stdout.strip().split('\n')
    return [f for f in files if f]

def extract_sensor_region(filepath):
    """Extract sensor and region from filepath."""
    parts = filepath.split('/')
    # Find sensor name (usually 3 levels after spectrum_bands_2)
    try:
        idx = parts.index('spectrum_bands_2')
        sensor = parts[idx + 1] if idx + 1 < len(parts) else 'unknown'
    except ValueError:
        sensor = 'unknown'
    
    # Extract region from filename
    filename = os.path.basename(filepath)
    # Try to extract country code from filename
    parts = filename.split('_')
    region = 'unknown'
    for part in parts:
        if len(part) == 2 or len(part) == 3:  # Country codes are usually 2-3 chars
            if part.lower() not in ['fm', 'tetra', 'dab', 'dvbt', 'gsm', 'lte', 'spectrumbands']:
                region = part
                break
    
    return sensor, region

def analyze_misclassifications():
    """Analyze misclassifications in detail."""
    print("=" * 70)
    print("Step 8: Detailed Misclassification Analysis")
    print("=" * 70)
    
    # Load models
    print("\nLoading models...")
    scaler, encoder, model = load_models()
    print("Models loaded")
    
    # Process FM files
    print("\n" + "=" * 70)
    print("FM ANALYSIS (41 files, ground truth: fm)")
    print("=" * 70)
    
    fm_files = find_files('fm')
    fm_predictions = []
    fm_confusion = Counter()
    fm_sensor_errors = Counter()
    fm_region_errors = Counter()
    
    for filepath in fm_files:
        filename = os.path.basename(filepath)
        data = np.load(filepath)[:50, :].astype(np.float32)
        features, _ = extract_33_features(data)
        predicted_label, _, _ = classify(features, scaler, encoder, model)
        
        sensor, region = extract_sensor_region(filepath)
        
        if predicted_label != 'fm':
            fm_confusion[predicted_label] += 1
            fm_sensor_errors[sensor] += 1
            fm_region_errors[region] += 1
        
        fm_predictions.append({
            'file': filename,
            'predicted': predicted_label,
            'correct': predicted_label == 'fm',
            'sensor': sensor,
            'region': region
        })
    
    print(f"\nFM Misclassification Summary:")
    print(f"  Total files: {len(fm_files)}")
    print(f"  Correct: {sum(1 for p in fm_predictions if p['correct'])}")
    print(f"  Incorrect: {sum(1 for p in fm_predictions if not p['correct'])}")
    
    print(f"\n  Confused with (predicted as):")
    for tech, count in fm_confusion.most_common():
        print(f"    {tech:6s}: {count:3d} times ({count/len(fm_files)*100:.1f}%)")
    
    print(f"\n  Top sensors with errors:")
    for sensor, count in fm_sensor_errors.most_common(5):
        print(f"    {sensor:30s}: {count:3d} errors")
    
    print(f"\n  Top regions with errors:")
    for region, count in fm_region_errors.most_common(5):
        print(f"    {region:10s}: {count:3d} errors")
    
    # Process TETRA files
    print("\n" + "=" * 70)
    print("TETRA ANALYSIS (38 files, ground truth: tetra)")
    print("=" * 70)
    
    tetra_files = find_files('tetra')
    tetra_predictions = []
    tetra_confusion = Counter()
    tetra_correct_sensors = []
    tetra_sensor_errors = Counter()
    tetra_region_errors = Counter()
    
    for filepath in tetra_files:
        filename = os.path.basename(filepath)
        data = np.load(filepath)[:50, :].astype(np.float32)
        features, _ = extract_33_features(data)
        predicted_label, _, _ = classify(features, scaler, encoder, model)
        
        sensor, region = extract_sensor_region(filepath)
        
        if predicted_label == 'tetra':
            tetra_correct_sensors.append(sensor)
        else:
            tetra_confusion[predicted_label] += 1
            tetra_sensor_errors[sensor] += 1
            tetra_region_errors[region] += 1
        
        tetra_predictions.append({
            'file': filename,
            'predicted': predicted_label,
            'correct': predicted_label == 'tetra',
            'sensor': sensor,
            'region': region
        })
    
    print(f"\nTETRA Misclassification Summary:")
    print(f"  Total files: {len(tetra_files)}")
    print(f"  Correct: {sum(1 for p in tetra_predictions if p['correct'])}")
    print(f"  Incorrect: {sum(1 for p in tetra_predictions if not p['correct'])}")
    
    print(f"\n  Confused with (predicted as):")
    for tech, count in tetra_confusion.most_common():
        print(f"    {tech:6s}: {count:3d} times ({count/len(tetra_files)*100:.1f}%)")
    
    print(f"\n  Correctly classified sensors:")
    for sensor in set(tetra_correct_sensors):
        count = tetra_correct_sensors.count(sensor)
        print(f"    {sensor:30s}: {count:3d} correct")
    
    print(f"\n  Top sensors with errors:")
    for sensor, count in tetra_sensor_errors.most_common(5):
        print(f"    {sensor:30s}: {count:3d} errors")
    
    print(f"\n  Top regions with errors:")
    for region, count in tetra_region_errors.most_common(5):
        print(f"    {region:10s}: {count:3d} errors")
    
    # Combined analysis
    print("\n" + "=" * 70)
    print("COMBINED ANALYSIS")
    print("=" * 70)
    
    all_predictions = fm_predictions + tetra_predictions
    
    print(f"\nConfusion Matrix (Full Band):")
    print(f"{'':8s} {'dab':6s} {'dvbt':6s} {'fm':6s} {'gsm':6s} {'lte':6s} {'tetra':6s}")
    
    # Build confusion matrix
    confusion = {}
    for gt in ['fm', 'tetra']:
        confusion[gt] = Counter()
        for p in all_predictions:
            if (gt == 'fm' and p['file'].find('_fm_') != -1) or \
               (gt == 'tetra' and p['file'].find('_tetra_') != -1):
                confusion[gt][p['predicted']] += 1
    
    for gt in ['fm', 'tetra']:
        row = f"{gt:8s}"
        for pred in ['dab', 'dvbt', 'fm', 'gsm', 'lte', 'tetra']:
            count = confusion[gt].get(pred, 0)
            row += f"{count:6d}"
        print(row)
    
    print(f"\nKey Insights:")
    print(f"  1. FM is completely misclassified - mostly as GSM ({fm_confusion.get('gsm', 0)}/41)")
    print(f"  2. TETRA has some correct predictions ({sum(1 for p in tetra_predictions if p['correct'])}/38)")
    print(f"  3. TETRA is mostly confused with LTE ({tetra_confusion.get('lte', 0)}/38)")
    print(f"  4. Narrower bandwidth (TETRA: 538 bins) performs better than wider (FM: 2150 bins)")
    
    return fm_predictions, tetra_predictions

if __name__ == "__main__":
    fm_predictions, tetra_predictions = analyze_misclassifications()
