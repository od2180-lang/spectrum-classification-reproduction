#!/usr/bin/env python3
"""
Phase 2b: Signal Detection + Metadata Gate + Classification
Detect individual transmissions, use frequency metadata to gate, then classify.
"""

import sys
import os
import re
import math
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'PSD-technology-classification-framework')
from TDPackage.DetectorManager.Detector import ChannelDetector
from feature_extraction import extract_33_features
from inference import load_models, classify, TECH_LABELS

# Detection parameters (from author's code)
NOISE_PERCENTILE = 30
NOISE_SIGMA = 3
NOISE_THRES = 5
CV = 1
SMOOTHING = True
K = 0.2
WIDTH_APPLICABLE = True
PEAK_THRES = 3
DEF_NUM_TIMESEGMENTS = 50
MIN_TX_WIDTH = 5


def get_detection_params(start_freq):
    """Return detection parameters based on frequency band.
    
    Args:
        start_freq: Start frequency in MHz
        
    Returns:
        Dict with keys: distance, box_pts, peakThres, min_width
    """
    if 300 <= start_freq <= 430:
        return {'distance': 2, 'box_pts': 2, 'peakThres': 1, 'min_width': 2}
    elif 730 <= start_freq <= 830:
        return {'distance': 50, 'box_pts': 16, 'peakThres': 5, 'min_width': 10}
    else:
        return {'distance': 10, 'box_pts': 8, 'peakThres': 3, 'min_width': 10}


def calc_entropy(probs):
    """Compute Shannon entropy from probability array.
    
    Args:
        probs: Array of probabilities (should sum to 1)
        
    Returns:
        Entropy in bits (0 to log2(n_classes))
    """
    my_sum = 0
    for p in probs:
        if p > 0:
            my_sum += p * math.log(p, 2)
    return -my_sum


def parse_frequency(filename):
    """Extract start and end frequency (MHz) from filename.
    
    Handles both:
        SpectrumBands_389_394_tetra_...
        alcorcon_Feb_3_21SpectrumBands_421_426_tetra_...
    
    Returns:
        (start_freq, end_freq) tuple, or (0, 0) if parse fails
    """
    match = re.search(r'SpectrumBands_(\d+)_(\d+)_', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0


def map_frequency_to_tech(start_freq):
    """Map frequency to expected technology using expanded ranges for our dataset.
    
    Args:
        start_freq: Start frequency in MHz
        
    Returns:
        Expected technology string, or 'unkn' if unknown band
    """
    if 70 <= start_freq <= 130:
        return 'fm'
    elif 170 <= start_freq <= 240:
        return 'dab'
    elif 300 <= start_freq <= 430:
        return 'tetra'
    elif 460 <= start_freq <= 790:
        return 'dvbt'
    elif 730 <= start_freq <= 830:
        return 'lte'
    elif 910 <= start_freq <= 961:
        return 'gsm'
    else:
        return 'unkn'


def noise_estimation(data):
    """Estimate noise floor using percentile fit (approximates author's define_noise_level)."""
    noise_values = data[data < np.percentile(data, NOISE_PERCENTILE)]
    noise_db = np.mean(noise_values) + NOISE_SIGMA * np.std(noise_values)
    return noise_db


def detect_transmissions(data, noise_db, start_freq=0):
    """Detect transmissions using author's ChannelDetector with adaptive params."""
    params = get_detection_params(start_freq)
    detector = ChannelDetector(NOISE_THRES, CV, SMOOTHING, K, WIDTH_APPLICABLE,
                               distance=params['distance'], box_pts=params['box_pts'])
    detector.data = data
    detector.tx_detection_funct(noise_db, noiseThres=NOISE_THRES, peakThres=params['peakThres'])
    if detector.channels_detected is not None:
        detector.channels_detected = np.array([
            ch for ch in detector.channels_detected
            if int(ch[1]) - int(ch[0]) + 1 >= params['min_width']
        ])
    return detector.channels_detected


def extract2MHz(dta):
    """Center-crop to 200 bins for wideband signals (author's extract2MHz)."""
    width = dta.shape[1]
    center = round(width / 2)
    if center < 101:
        return None
    return dta[:, (center - 100):(center + 100)]


def width_gate(expected_tech, width):
    """Author's metadata width gates. Returns True if signal should be classified."""
    if expected_tech == 'tetra':
        return width < 10
    elif expected_tech == 'gsm':
        return 14 <= width <= 35
    elif expected_tech == 'dab':
        return 120 <= width <= 240
    elif expected_tech == 'dvbt':
        return width >= 400
    elif expected_tech == 'lte':
        return width > 700
    else:
        return True


def process_transmission(data, start, end, scaler, encoder, model, expected_tech=None):
    """Classify one detected transmission with center crop and width gates."""
    try:
        tx = data[:, start:end + 1]
        width = tx.shape[1]
        if width < MIN_TX_WIDTH:
            return None, "too_narrow", 0.0

        # Author's width gate
        if expected_tech and not width_gate(expected_tech, width):
            return None, "width_gate_fail", 0.0

        # Center crop to 200 bins for wideband signals
        if width >= 200:
            tx_cropped = extract2MHz(tx)
            if tx_cropped is None:
                return None, "crop_fail", 0.0
            tx = tx_cropped

        tx = tx[:DEF_NUM_TIMESEGMENTS, :]
        tx = tx.astype(np.float32)
        if np.all(tx == tx[0]):
            return None, "flat_signal", 0.0
        features, _ = extract_33_features(tx)
        features = features.astype(float)
        features[~np.isfinite(features)] = 0
        if features.shape[0] == 0:
            return None, "no_features", 0.0
        scaled = scaler.transform(features)
        encoded = encoder.predict(scaled, verbose=0)
        reshaped = encoded.reshape(-1, 16, 1)
        preds = model.predict(reshaped, verbose=0)
        avg_preds = np.mean(preds, axis=0)
        entropies = [calc_entropy(row) for row in preds]
        entropy_avg = np.mean(entropies)
        if entropy_avg > 0.7:
            return 'unkn', None, entropy_avg
        predicted_class = np.argmax(avg_preds)
        predicted_label = TECH_LABELS[predicted_class]
        return predicted_label, None, entropy_avg
    except Exception as e:
        return None, f"error: {str(e)}", 0.0


def process_file(file_path, ground_truth, scaler, encoder, model):
    """Process one file: detect transmissions and classify each one."""
    results = []
    
    fname = os.path.basename(file_path)
    start_freq, end_freq = parse_frequency(fname)
    expected_tech = map_frequency_to_tech(start_freq)
    
    try:
        data = np.load(file_path, allow_pickle=True)
        noise_db = noise_estimation(data)
        channels = detect_transmissions(data, noise_db, start_freq)

        if channels is None:
            return results, start_freq, end_freq

        for ch in channels:
            start, end = int(ch[0]), int(ch[1])
            predicted, error, entropy = process_transmission(data, start, end, scaler, encoder, model, expected_tech)
            width = end - start + 1
            results.append({
                'predicted': predicted,
                'expected': expected_tech,
                'ground_truth': ground_truth,
                'width': width,
                'error': error,
                'entropy': entropy,
                'match': predicted == expected_tech if predicted else False
            })

    except Exception as e:
        print(f"  Error loading {fname}: {e}")
    
    return results, start_freq, end_freq


def main():
    import subprocess
    from collections import Counter

    tech_patterns = {
        'fm': '*_fm_*.npy',
        'tetra': '*_tetra_*.npy',
        'lte': '*_lte_*.npy',
        'gsm': '*_gsm_*.npy',
        'dab': '*_dab_*.npy',
        'dvbt': '*_dvbt_*.npy',
    }

    tech_files = {}
    for tech, pattern in tech_patterns.items():
        result = subprocess.run(['find', 'dataset', '-name', pattern], capture_output=True, text=True)
        files = sorted(result.stdout.strip().split('\n'))
        tech_files[tech] = [f for f in files if f]
        print(f"  {tech.upper()}: {len(tech_files[tech])} files")

    print("Loading models...")
    scaler, encoder, model = load_models()
    print("Models loaded\n")

    all_tech_results = {}
    all_tech_file_results = {}

    for tech in ['fm', 'tetra', 'lte', 'gsm', 'dab', 'dvbt']:
        files = tech_files[tech]
        if not files:
            print(f"\n{'=' * 60}")
            print(f"Processing {tech.upper()} files... (0 files)")
            print("=" * 60)
            all_tech_results[tech] = []
            all_tech_file_results[tech] = []
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing {tech.upper()} files... ({len(files)} files)")
        print("=" * 60)
        tech_results = []
        tech_file_results = []
        for i, fpath in enumerate(files):
            fname = os.path.basename(fpath)
            results, start_freq, end_freq = process_file(fpath, tech, scaler, encoder, model)
            print(f"  File {i + 1}/{len(files)}: {fname} ({start_freq}-{end_freq}MHz)")
            tech_results.extend(results)
            tech_file_results.append(results)
            for j, r in enumerate(results):
                if r['error']:
                    print(f"    TX {j + 1} ({r['width']} bins): {r['error']}")
                else:
                    mark = "+" if r['match'] else "-"
                    print(f"    TX {j + 1} ({r['width']} bins): {r['predicted']} ent={r['entropy']:.3f} {mark}")

        all_tech_results[tech] = tech_results
        all_tech_file_results[tech] = tech_file_results

    print("\n" + "=" * 60)
    print("RESULTS (per-transmission)")
    print("=" * 60)

    for tech in ['fm', 'tetra', 'lte', 'gsm', 'dab', 'dvbt']:
        results = all_tech_results[tech]
        correct = sum(1 for r in results if r['match'])
        total = sum(1 for r in results if not r['error'])
        if total > 0:
            print(f"  {tech.upper():>6}: {correct}/{total} = {correct / total * 100:.1f}%")
        else:
            print(f"  {tech.upper():>6}: 0/0")

    print("\n--- Per-file accuracy (majority vote) ---")
    for tech in ['fm', 'tetra', 'lte', 'gsm', 'dab', 'dvbt']:
        file_results = all_tech_file_results[tech]
        correct_files = 0
        total_files = len(file_results)
        for fres in file_results:
            valid_tx = [r for r in fres if not r['error']]
            if not valid_tx:
                continue
            correct_tx = sum(1 for r in valid_tx if r['match'])
            if correct_tx > len(valid_tx) / 2:
                correct_files += 1
        if total_files > 0:
            print(f"  {tech.upper():>6}: {correct_files}/{total_files} files = {correct_files / total_files * 100:.1f}%")

    all_results = []
    for tech in ['fm', 'tetra', 'lte', 'gsm', 'dab', 'dvbt']:
        all_results.extend(all_tech_results[tech])
    valid = [r for r in all_results if not r['error']]
    techs = ['dab', 'dvbt', 'fm', 'gsm', 'lte', 'tetra']

    print(f"\nConfusion Matrix (predicted vs expected):")
    print(f"{'':>10}", end="")
    for t in techs:
        print(f"{t:>8}", end="")
    print()

    for exp in techs:
        print(f"{exp:>10}", end="")
        for pred in techs:
            count = sum(1 for r in valid if r['expected'] == exp and r['predicted'] == pred)
            print(f"{count:>8}", end="")
        print()

    skipped = [r for r in all_results if r['error']]
    if skipped:
        error_counts = Counter(r['error'] for r in skipped)
        print(f"\nErrors ({len(skipped)} total):")
        for err, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {err}: {count}")

    entropies = [r['entropy'] for r in valid]
    if entropies:
        print(f"\nEntropy stats: min={min(entropies):.3f} max={max(entropies):.3f} avg={np.mean(entropies):.3f}")
        high_ent = [r for r in valid if r['entropy'] > 0.7]
        print(f"High entropy (>0.7): {len(high_ent)}/{len(valid)} transmissions")


if __name__ == "__main__":
    main()
