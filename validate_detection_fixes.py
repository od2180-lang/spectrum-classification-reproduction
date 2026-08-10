#!/usr/bin/env python3
"""
Validation script for detection width fixes.
Tests Config C (TETRA fix) and Config D (LTE fix) on subset of files.
"""

import os
import re
import sys
import math
import argparse
import numpy as np
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

warnings.filterwarnings('ignore')

sys.path.insert(0, 'PSD-technology-classification-framework')
from TDPackage.DetectorManager.Detector import ChannelDetector
from feature_extraction import extract_33_features
from inference import load_models, classify, TECH_LABELS

# Constants
DEF_NUM_TIMESEGMENTS = 50
MIN_TX_WIDTH = 5
NOISE_PERCENTILE = 30
NOISE_SIGMA = 3
NOISE_THRES = 5
CV = 1
SMOOTHING = True
K = 0.2
WIDTH_APPLICABLE = True

# Configurations
CONFIGS = {
    'baseline': {
        'tetra_noise': 'global',
        'tetra_params': {'distance': 2, 'box_pts': 2, 'peakThres': 1, 'min_width': 2},
        'lte_params': {'distance': 50, 'box_pts': 16, 'peakThres': 5, 'min_width': 10},
        'default_params': {'distance': 10, 'box_pts': 8, 'peakThres': 3, 'min_width': 10},
        'lte_pipeline': 'gate_then_crop',
        'lte_gate': 700,
        'tetra_gate': 10,
    },
    'A': {
        'tetra_noise': 'per_band',
        'tetra_params': {'distance': 2, 'box_pts': 2, 'peakThres': 1, 'min_width': 2},
        'lte_params': {'distance': 50, 'box_pts': 16, 'peakThres': 5, 'min_width': 10},
        'default_params': {'distance': 10, 'box_pts': 8, 'peakThres': 3, 'min_width': 10},
        'lte_pipeline': 'gate_then_crop',
        'lte_gate': 700,
        'tetra_gate': 10,
    },
    'B': {
        'tetra_noise': 'global',
        'tetra_params': {'distance': 1, 'box_pts': 1, 'peakThres': 0.5, 'min_width': 1},
        'lte_params': {'distance': 50, 'box_pts': 16, 'peakThres': 5, 'min_width': 10},
        'default_params': {'distance': 10, 'box_pts': 8, 'peakThres': 3, 'min_width': 10},
        'lte_pipeline': 'gate_then_crop',
        'lte_gate': 700,
        'tetra_gate': 10,
    },
    'C': {
        'tetra_noise': 'per_band',
        'tetra_params': {'distance': 1, 'box_pts': 1, 'peakThres': 0.5, 'min_width': 1},
        'lte_params': {'distance': 50, 'box_pts': 16, 'peakThres': 5, 'min_width': 10},
        'default_params': {'distance': 10, 'box_pts': 8, 'peakThres': 3, 'min_width': 10},
        'lte_pipeline': 'gate_then_crop',
        'lte_gate': 700,
        'tetra_gate': 10,
    },
    'D': {
        'tetra_noise': 'global',
        'tetra_params': {'distance': 2, 'box_pts': 2, 'peakThres': 1, 'min_width': 2},
        'lte_params': {'distance': 50, 'box_pts': 16, 'peakThres': 5, 'min_width': 10},
        'default_params': {'distance': 10, 'box_pts': 8, 'peakThres': 3, 'min_width': 10},
        'lte_pipeline': 'crop_then_gate',
        'lte_gate': 200,
        'tetra_gate': 10,
    },
}

def parse_frequency(filename):
    match = re.search(r'SpectrumBands_(\d+)_(\d+)_', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0

def map_frequency_to_tech(start_freq):
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

def noise_estimation_global(data):
    noise_values = data[data < np.percentile(data, NOISE_PERCENTILE)]
    return np.mean(noise_values) + NOISE_SIGMA * np.std(noise_values)

def noise_estimation_per_band(data, start_freq, end_freq, freq_resolution=9.3):
    total_bins = data.shape[1]
    data_start = start_freq - (total_bins * freq_resolution / 2)
    start_bin = max(0, int((start_freq - data_start) / freq_resolution))
    end_bin = min(total_bins, int((end_freq - data_start) / freq_resolution) + 1)
    if start_bin >= end_bin:
        return noise_estimation_global(data)
    band_data = data[:, start_bin:end_bin]
    noise_values = band_data[band_data < np.percentile(band_data, NOISE_PERCENTILE)]
    if len(noise_values) == 0:
        return noise_estimation_global(data)
    return np.mean(noise_values) + NOISE_SIGMA * np.std(noise_values)

def get_detection_params(config, start_freq):
    if 300 <= start_freq <= 430:
        return config['tetra_params']
    elif 730 <= start_freq <= 830:
        return config['lte_params']
    else:
        return config['default_params']

def detect_transmissions(data, noise_db, start_freq, config):
    params = get_detection_params(config, start_freq)
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
    width = dta.shape[1]
    center = round(width / 2)
    if center < 101:
        return None
    return dta[:, (center - 100):(center + 100)]

def width_gate(config, expected_tech, width):
    if expected_tech == 'tetra':
        return width < config['tetra_gate']
    elif expected_tech == 'gsm':
        return 14 <= width <= 35
    elif expected_tech == 'dab':
        return 120 <= width <= 240
    elif expected_tech == 'dvbt':
        return width >= 400
    elif expected_tech == 'lte':
        return width > config['lte_gate']
    else:
        return True

def calc_entropy(probs):
    my_sum = 0
    for p in probs:
        if p > 0:
            my_sum += p * math.log(p, 2)
    return -my_sum

def process_transmission(data, start, end, scaler, encoder, model, expected_tech, config, temperature):
    try:
        tx = data[:, start:end + 1]
        width = tx.shape[1]
        if width < MIN_TX_WIDTH:
            return None, "too_narrow", 0.0, width

        if config['lte_pipeline'] == 'gate_then_crop':
            if expected_tech and not width_gate(config, expected_tech, width):
                return None, "width_gate_fail", 0.0, width
            if width >= 200:
                tx_cropped = extract2MHz(tx)
                if tx_cropped is None:
                    return None, "crop_fail", 0.0, width
                tx = tx_cropped
        else:
            if width >= 200:
                tx_cropped = extract2MHz(tx)
                if tx_cropped is None:
                    return None, "crop_fail", 0.0, width
                tx = tx_cropped
                width = tx.shape[1]
            if expected_tech and not width_gate(config, expected_tech, width):
                return None, "width_gate_fail", 0.0, width

        tx = tx[:DEF_NUM_TIMESEGMENTS, :]
        tx = tx.astype(np.float32)
        if np.all(tx == tx[0]):
            return None, "flat_signal", 0.0, width
        features, _ = extract_33_features(tx)
        features = features.astype(float)
        features[~np.isfinite(features)] = 0
        if features.shape[0] == 0:
            return None, "no_features", 0.0, width
        
        predicted_label, predicted_class, avg_predictions = classify(features, scaler, encoder, model, temperature=temperature)
        
        entropies = [calc_entropy(row) for row in np.array([avg_predictions])]
        entropy_avg = np.mean(entropies)
        if entropy_avg > 0.7:
            return 'unkn', None, entropy_avg, width
        
        return predicted_label, None, entropy_avg, width
    except Exception as e:
        return None, f"error: {str(e)}", 0.0, width

def process_file(args):
    file_path, ground_truth, config_name, temperature, scaler, encoder, model = args
    config = CONFIGS[config_name]
    results = []
    
    fname = os.path.basename(file_path)
    start_freq, end_freq = parse_frequency(fname)
    expected_tech = map_frequency_to_tech(start_freq)
    
    try:
        data = np.load(file_path, allow_pickle=True)
        
        if config['tetra_noise'] == 'per_band' and expected_tech == 'tetra':
            noise_db = noise_estimation_per_band(data, start_freq, end_freq)
        else:
            noise_db = noise_estimation_global(data)
        
        channels = detect_transmissions(data, noise_db, start_freq, config)
        
        if channels is not None:
            for ch in channels:
                start, end = int(ch[0]), int(ch[1])
                predicted, error, entropy, width = process_transmission(
                    data, start, end, scaler, encoder, model, expected_tech, config, temperature
                )
                results.append({
                    'config': config_name,
                    'tech': ground_truth,
                    'file': fname,
                    'width': width,
                    'gate_pass': error != "width_gate_fail",
                    'predicted': predicted,
                    'expected': expected_tech,
                    'ground_truth': ground_truth,
                    'match': predicted == expected_tech if predicted else False,
                    'error': error,
                    'entropy': entropy,
                    'temperature': temperature,
                })
    except Exception as e:
        results.append({
            'config': config_name,
            'tech': ground_truth,
            'file': fname,
            'width': 0,
            'gate_pass': False,
            'predicted': None,
            'expected': expected_tech,
            'ground_truth': ground_truth,
            'match': False,
            'error': f"load_error: {str(e)}",
            'entropy': 0.0,
            'temperature': temperature,
        })
    
    return results

def find_test_files():
    import subprocess
    tech_patterns = {
        'tetra': '*_tetra_*.npy',
        'lte': '*_lte_*.npy',
        'fm': '*_fm_*.npy',
    }
    found = {tech: [] for tech in tech_patterns}
    base = 'dataset/opt/shared/alessio/Data_Transmitters_Identification/spectrum_bands_2'
    
    for tech, pattern in tech_patterns.items():
        result = subprocess.run(['find', base, '-name', pattern], capture_output=True, text=True)
        files = result.stdout.strip().split('\n')
        files = [f for f in files if f]
        found[tech] = files[:3] if tech != 'fm' else files[:2]
    
    return found

def print_summary(all_results):
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    for config_name in ['baseline', 'A', 'B', 'C', 'D']:
        config_results = [r for r in all_results if r['config'] == config_name]
        if not config_results:
            continue
        
        print(f"\n--- Config {config_name} ---")
        
        for tech in ['tetra', 'lte', 'fm']:
            tech_results = [r for r in config_results if r['tech'] == tech]
            if not tech_results:
                continue
            
            widths = [r['width'] for r in tech_results if r['width'] > 0]
            gate_passed = [r for r in tech_results if r['gate_pass']]
            valid = [r for r in tech_results if r['error'] is None or r['error'] == '']
            
            print(f"  {tech.upper()}: {len(tech_results)} detections, {len(gate_passed)} passed gate")
            if widths:
                print(f"    Widths: min={min(widths)}, max={max(widths)}, median={np.median(widths):.1f}")
            if gate_passed:
                correct = sum(1 for r in gate_passed if r['match'])
                print(f"    Accuracy (gated): {correct}/{len(gate_passed)} = {correct/len(gate_passed)*100:.1f}%")
                entropies = [r['entropy'] for r in gate_passed if r['entropy'] > 0]
                if entropies:
                    print(f"    Entropy: mean={np.mean(entropies):.3f}, max={max(entropies):.3f}")

def main():
    parser = argparse.ArgumentParser(description='Validate detection width fixes')
    parser.add_argument('--configs', nargs='+', default=['baseline', 'C', 'D'],
                        choices=['baseline', 'A', 'B', 'C', 'D'],
                        help='Configurations to test')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers')
    parser.add_argument('--temperature', type=float, default=2.0, help='Temperature scaling')
    parser.add_argument('--output', default='validation_results.csv', help='Output CSV')
    args = parser.parse_args()
    
    print("Loading models...")
    scaler, encoder, model = load_models()
    print("Models loaded.")
    
    test_files = find_test_files()
    print(f"\nFound files:")
    for tech, files in test_files.items():
        print(f"  {tech}: {len(files)} files")
        for f in files:
            print(f"    {os.path.basename(f)}")
    
    task_args = []
    for tech, files in test_files.items():
        for f in files:
            for config_name in args.configs:
                task_args.append((f, tech, config_name, args.temperature, scaler, encoder, model))
    
    print(f"\nRunning {len(task_args)} tasks with {args.workers} workers...")
    
    all_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file, arg): arg for arg in task_args}
        for future in as_completed(futures):
            results = future.result()
            all_results.extend(results)
            fname = futures[future][0]
            config = futures[future][2]
            tech = futures[future][1]
            print(f"  Done: {tech}/{config} - {os.path.basename(fname)}")
    
    import csv
    if all_results:
        keys = all_results[0].keys()
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {args.output}")
    
    print_summary(all_results)
    
    print("\n" + "=" * 80)
    print("GO/NO-GO DECISION")
    print("=" * 80)
    
    for config_name in ['C', 'D']:
        if config_name not in args.configs:
            continue
        config_results = [r for r in all_results if r['config'] == config_name]
        tetra_results = [r for r in config_results if r['tech'] == 'tetra']
        lte_results = [r for r in config_results if r['tech'] == 'lte']
        
        if config_name == 'C':
            widths = [r['width'] for r in tetra_results if r['width'] > 0]
            gate_passed = [r for r in tetra_results if r['gate_pass']]
            mode_width = max(set(widths), key=widths.count) if widths else 0
            pass_rate = len(gate_passed) / len(tetra_results) if tetra_results else 0
            
            print(f"\nConfig C (TETRA fix):")
            print(f"  Mode width: {mode_width} bins (target: 2-5)")
            print(f"  Gate pass rate: {pass_rate*100:.1f}% (target: >50%)")
            if mode_width <= 5 and pass_rate > 0.5:
                print("  >>> GO for full run")
            else:
                print("  >>> NO-GO - needs iteration")
        
        elif config_name == 'D':
            gate_passed = [r for r in lte_results if r['gate_pass']]
            pass_rate = len(gate_passed) / len(lte_results) if lte_results else 0
            
            print(f"\nConfig D (LTE fix):")
            print(f"  Gate pass rate: {pass_rate*100:.1f}% (target: >30%)")
            if pass_rate > 0.3:
                print("  >>> GO for full run")
            else:
                print("  >>> NO-GO - needs iteration")

if __name__ == '__main__':
    main()