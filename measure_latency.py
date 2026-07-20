#!/usr/bin/env python3
"""
Measure classification latency: full bandwidth vs 2 MHz section.
Recreates Figure 15 (TCS latency per technology) from the paper.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = os.path.join(BASE_DIR, 'PSD-technology-classification-framework')
sys.path.insert(0, FRAMEWORK_DIR)
sys.path.insert(0, BASE_DIR)

from run_classification import load_models, detect_transmissions, extract_features, extract_transmission, LABELS

FILES = {
    'DVB-T': os.path.join(BASE_DIR, 'spectrum_bands/BZ_Italy/Sep_1/SpectrumBands_540_649_dvbt_Ita_540_649.npy'),
    'GSM': os.path.join(BASE_DIR, 'spectrum_bands/Nudelsalat_RPi/Aug_1/SpectrumBands_925_960_gsm_Ger_925_960.npy'),
}

TARGET_BINS = 200
NUM_WARMUP = 1

def classify_full_pipeline(tx_data, encoder, classifier, scaler):
    features = extract_features(tx_data)
    features_2d = features.reshape(1, -1)
    features_scaled = scaler.transform(features_2d)
    features_encoded = encoder.predict(features_scaled, verbose=0)
    features_lstm = features_encoded.reshape(1, 16, 1)
    predictions = classifier.predict(features_lstm, verbose=0)
    pred_class = np.argmax(predictions[0])
    confidence = predictions[0][pred_class]
    probs = predictions[0] + 1e-10
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(probs))
    normalized_entropy = entropy / max_entropy
    if normalized_entropy > 0.7:
        return 'unkn', confidence, normalized_entropy
    return LABELS[pred_class], confidence, normalized_entropy


def time_pipeline_full_vs_2mhz(spectrogram, encoder, classifier, scaler):
    transmissions = detect_transmissions(spectrogram)
    full_latencies, mhz2_latencies = [], []
    for start_bin, end_bin in transmissions:
        tx_full = spectrogram[:, start_bin:end_bin]
        tx_full = tx_full[:50, :]
        if tx_full.size < 10:
            continue
        start = time.perf_counter()
        classify_full_pipeline(tx_full, encoder, classifier, scaler)
        full_latencies.append(time.perf_counter() - start)
        tx_2mhz = extract_transmission(spectrogram, start_bin, end_bin, target_bins=TARGET_BINS)
        if tx_2mhz.size < 10:
            continue
        start = time.perf_counter()
        classify_full_pipeline(tx_2mhz, encoder, classifier, scaler)
        mhz2_latencies.append(time.perf_counter() - start)
    return np.array(full_latencies)*1000, np.array(mhz2_latencies)*1000


def main():
    print("=" * 60)
    print("  Latency Measurement: Full Bandwidth vs 2 MHz")
    print("  Recreating Figure 15")
    print("=" * 60)
    print()

    print("Loading models...")
    encoder, classifier, scaler = load_models()
    print()

    dummy = np.random.randn(50, 200).astype(np.float32)
    for _ in range(NUM_WARMUP):
        classify_full_pipeline(dummy, encoder, classifier, scaler)
    print("Warmup complete")
    print()

    results = {}
    for tech_name, filepath in FILES.items():
        print(f"Processing {tech_name}...")
        print(f"  File: {os.path.basename(filepath)}")
        spectrogram = np.load(filepath, allow_pickle=True)
        print(f"  Shape: {spectrogram.shape}")
        if spectrogram.shape[0] > 100:
            spectrogram = spectrogram[:100, :]
        full_lat, mhz2_lat = time_pipeline_full_vs_2mhz(spectrogram, encoder, classifier, scaler)
        print(f"  Transmissions detected: {len(full_lat)}")
        print(f"  Full BW mean: {np.mean(full_lat):.1f} ms (±{np.std(full_lat):.1f})")
        print(f"  2 MHz mean:  {np.mean(mhz2_lat):.1f} ms (±{np.std(mhz2_lat):.1f})")
        results[tech_name] = {'full': full_lat, '2mhz': mhz2_lat}
        print()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    techs = list(results.keys())
    x = np.arange(len(techs))
    w = 0.35
    ax.bar(x - w/2, [results[t]['full'].mean() for t in techs], w, label='Full Bandwidth', color='#1f77b4', edgecolor='black')
    ax.bar(x + w/2, [results[t]['2mhz'].mean() for t in techs], w, label='2 MHz Section', color='#4a006e', edgecolor='black')
    ax.set_xlabel('Wireless Technology', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classification Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('TCS Latency: Full Bandwidth vs 2 MHz Section', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(techs, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    for i, t in enumerate(techs):
        ax.text(i - 0.35/2, results[t]['full'].mean() + 1, f"{results[t]['full'].mean():.1f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(i + 0.35/2, results[t]['2mhz'].mean() + 1, f"{results[t]['2mhz'].mean():.1f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(0, max(max(r['full'].mean(), r['2mhz'].mean()) for r in results.values()) * 1.3)
    plt.tight_layout()

    output_dir = os.path.join(BASE_DIR, 'figures')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'figure_15_latency_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    print("Done!")


if __name__ == '__main__':
    main()