#!/usr/bin/env python3
"""
Phase 3: Quick Parameter Test (Reduced Dataset)
Test on 5 FM + 5 TETRA files for speed.
"""

import sys
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'PSD-technology-classification-framework')
from TDPackage.DetectorManager.Detector import ChannelDetector
from feature_extraction import extract_33_features
from inference import load_models, classify, TECH_LABELS
import subprocess


def noise_estimation(data, percentile=30, sigma=3):
    noise_values = data[data < np.percentile(data, percentile)]
    return np.mean(noise_values) + sigma * np.std(noise_values)


def test_params(files, truth, scaler, encoder, model, params):
    correct = 0
    total = 0
    
    for fpath in files:
        try:
            data = np.load(fpath, allow_pickle=True)
            noise_db = noise_estimation(data, params.get('pct', 30), params.get('sigma', 3))
            
            detector = ChannelDetector(params['nthres'], params['cv'], True, params['k'], True)
            detector.data = data
            detector.tx_detection_funct(noise_db, noiseThres=params['nthres'], peakThres=params['pthres'])
            
            if detector.channels_detected is None:
                continue
            
            for ch in detector.channels_detected:
                start, end = int(ch[0]), int(ch[1])
                width = end - start + 1
                if width < params['minw']:
                    continue
                
                tx = data[:, start:end + 1][:50, :].astype(np.float32)
                if np.all(tx == tx[0]):
                    continue
                
                features, _ = extract_33_features(tx)
                features = features.astype(float)
                if np.any(np.isnan(features)):
                    continue
                
                pred, _, _ = classify(features, scaler, encoder, model)
                total += 1
                if pred == truth:
                    correct += 1
        except Exception:
            continue
    
    acc = correct / total * 100 if total > 0 else 0
    return acc, total


def main():
    print("Loading models...")
    scaler, encoder, model = load_models()
    
    # Use subset of files for speed
    result = subprocess.run(['find', 'dataset', '-name', '*_fm_*.npy'], capture_output=True, text=True)
    fm_all = sorted(result.stdout.strip().split('\n'))
    fm_files = fm_all[:5]
    
    result = subprocess.run(['find', 'dataset', '-name', '*_tetra_*.npy'], capture_output=True, text=True)
    te_all = sorted(result.stdout.strip().split('\n'))
    te_files = te_all[:5]
    
    print(f"Using {len(fm_files)} FM, {len(te_files)} TETRA files\n")
    
    baseline = {'nthres': 5, 'cv': 1, 'k': 0.2, 'pthres': 3, 'minw': 5}
    
    # Get baseline
    fm_acc, fm_t = test_params(fm_files, 'fm', scaler, encoder, model, baseline)
    te_acc, te_t = test_params(te_files, 'tetra', scaler, encoder, model, baseline)
    print(f"BASELINE: FM={fm_acc:.1f}% ({fm_t} TXs), TETRA={te_acc:.1f}% ({te_t} TXs)\n")
    
    # Experiment 1: Peak Threshold
    print("=" * 60)
    print("EXP 1: Peak Threshold")
    print("=" * 60)
    for pthres in [1, 2, 3, 5, 7, 10]:
        p = baseline.copy()
        p['pthres'] = pthres
        fm_a, fm_n = test_params(fm_files, 'fm', scaler, encoder, model, p)
        te_a, te_n = test_params(te_files, 'tetra', scaler, encoder, model, p)
        print(f"  pthres={pthres:<4} FM={fm_a:.1f}%({fm_n}tx) TETRA={te_a:.1f}%({te_n}tx)")
    
    # Experiment 2: Noise Threshold
    print("\n" + "=" * 60)
    print("EXP 2: Noise Threshold")
    print("=" * 60)
    for nthres in [3, 5, 7, 10, 15]:
        p = baseline.copy()
        p['nthres'] = nthres
        fm_a, fm_n = test_params(fm_files, 'fm', scaler, encoder, model, p)
        te_a, te_n = test_params(te_files, 'tetra', scaler, encoder, model, p)
        print(f"  nthres={nthres:<4} FM={fm_a:.1f}%({fm_n}tx) TETRA={te_a:.1f}%({te_n}tx)")
    
    # Experiment 3: K Factor
    print("\n" + "=" * 60)
    print("EXP 3: K Factor")
    print("=" * 60)
    for k in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        p = baseline.copy()
        p['k'] = k
        fm_a, fm_n = test_params(fm_files, 'fm', scaler, encoder, model, p)
        te_a, te_n = test_params(te_files, 'tetra', scaler, encoder, model, p)
        print(f"  k={k:<5} FM={fm_a:.1f}%({fm_n}tx) TETRA={te_a:.1f}%({te_n}tx)")
    
    # Experiment 4: Min Width
    print("\n" + "=" * 60)
    print("EXP 4: Minimum Width")
    print("=" * 60)
    for minw in [5, 8, 10, 15, 20]:
        p = baseline.copy()
        p['minw'] = minw
        fm_a, fm_n = test_params(fm_files, 'fm', scaler, encoder, model, p)
        te_a, te_n = test_params(te_files, 'tetra', scaler, encoder, model, p)
        print(f"  minw={minw:<4} FM={fm_a:.1f}%({fm_n}tx) TETRA={te_a:.1f}%({te_n}tx)")
    
    # Experiment 5: Noise Percentile
    print("\n" + "=" * 60)
    print("EXP 5: Noise Percentile")
    print("=" * 60)
    for pct in [10, 20, 30, 40, 50]:
        p = baseline.copy()
        p['pct'] = pct
        fm_a, fm_n = test_params(fm_files, 'fm', scaler, encoder, model, p)
        te_a, te_n = test_params(te_files, 'tetra', scaler, encoder, model, p)
        print(f"  pct={pct:<4} FM={fm_a:.1f}%({fm_n}tx) TETRA={te_a:.1f}%({te_n}tx)")


if __name__ == "__main__":
    main()
