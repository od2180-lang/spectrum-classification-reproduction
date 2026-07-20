#!/usr/bin/env python3
"""
Spectrum Classification Framework - Standalone Script
Replicates the classification pipeline from:
"A Framework for Wireless Technology Classification using Crowdsensing Platforms"
(IEEE INFOCOM 2023)
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SECTION 1: Configuration
# ============================================================

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = os.path.join(BASE_DIR, 'PSD-technology-classification-framework')
DATA_DIR = os.path.join(BASE_DIR, 'spectrum_bands')

# Model paths
SCALER_PATH = os.path.join(FRAMEWORK_DIR, 'TCpackage/resources/scaler/_AE16_LSTM_Scaler_.save')
ENCODER_PATH = os.path.join(FRAMEWORK_DIR, 'TCpackage/resources/save-DL-models/Autoencoder_DNN/TrainAllSensorHop_16_feat_mse_relu/saved-model-49-0.0002.hdf5')
CLASSIFIER_PATH = os.path.join(FRAMEWORK_DIR, 'TCpackage/resources/save-DL-models/LSTM_TrainWithAE/TrainAllSensorHop_DNNAE16_LSTM_mse_relu/saved-model-110-0.97.hdf5')

# Classification labels
LABELS = {0: 'dab', 1: 'dvbt', 2: 'fm', 3: 'gsm', 4: 'lte', 5: 'tetra', 6: 'unkn'}

# Feature names (33 tsfresh features)
FEATURE_NAMES = [
    'abs_energy', 'absolute_sum_of_changes', 'benford_correlation', 'cid_ce',
    'count_above_mean', 'count_below_mean', 'first_location_of_maximum',
    'first_location_of_minimum', 'has_duplicate', 'has_duplicate_max',
    'has_duplicate_min', 'kurtosis', 'last_location_of_maximum',
    'last_location_of_minimum', 'longest_strike_above_mean',
    'longest_strike_below_mean', 'maximum', 'mean', 'mean_abs_change',
    'mean_change', 'mean_second_derivative_central', 'median', 'minimum',
    'number_cwt_peaks', 'number_peaks', 'quantile', 'root_mean_square',
    'skewness', 'standard_deviation', 'sum_of_reoccurring_values',
    'sum_values', 'variance', 'variation_coefficient'
]


# ============================================================
# SECTION 2: Feature Extraction (33 tsfresh features)
# ============================================================

def extract_features(data):
    """
    Extract 33 statistical features from a 2D spectrogram using tsfresh.
    
    Args:
        data: 2D numpy array (time_segments x frequency_bins)
    
    Returns:
        1D array of 33 features
    """
    from tsfresh.feature_extraction.feature_calculators import (
        abs_energy, absolute_sum_of_changes, benford_correlation, cid_ce,
        count_above_mean, count_below_mean, first_location_of_maximum,
        first_location_of_minimum, has_duplicate, has_duplicate_max,
        has_duplicate_min, kurtosis, last_location_of_maximum,
        last_location_of_minimum, longest_strike_above_mean,
        longest_strike_below_mean, maximum, mean, mean_abs_change,
        mean_change, mean_second_derivative_central, median, minimum,
        number_cwt_peaks, number_peaks, quantile, root_mean_square,
        skewness, standard_deviation, sum_of_reoccurring_values,
        sum_values, variance, variation_coefficient
    )
    
    features = []
    
    # Average over time segments to get 1D profile
    if data.ndim == 2:
        profile = np.mean(data, axis=0)
    else:
        profile = data
    
    # Remove NaN and inf
    profile = np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Extract each feature
    feature_funcs = [
        abs_energy, absolute_sum_of_changes, benford_correlation, cid_ce,
        count_above_mean, count_below_mean, first_location_of_maximum,
        first_location_of_minimum, has_duplicate, has_duplicate_max,
        has_duplicate_min, kurtosis, last_location_of_maximum,
        last_location_of_minimum, longest_strike_above_mean,
        longest_strike_below_mean, maximum, mean, mean_abs_change,
        mean_change, mean_second_derivative_central, median, minimum,
        lambda x: number_cwt_peaks(x, n=3), lambda x: number_peaks(x, n=3),
        lambda x: quantile(x, 0.5), root_mean_square,
        skewness, standard_deviation, sum_of_reoccurring_values,
        sum_values, variance, variation_coefficient
    ]
    
    for func in feature_funcs:
        try:
            val = func(profile)
            if np.isnan(val) or np.isinf(val):
                val = 0.0
            features.append(float(val))
        except Exception:
            features.append(0.0)
    
    return np.array(features, dtype=np.float32)


# ============================================================
# SECTION 3: Transmission Detection
# ============================================================

def detect_transmissions(spectrogram, noise_percentile=10):
    """
    Detect transmissions in a spectrogram using energy detection.
    
    Args:
        spectrogram: 2D numpy array (time x frequency_bins)
        noise_percentile: percentile to estimate noise floor
    
    Returns:
        List of (start_bin, end_bin) tuples for each detected transmission
    """
    # Average over time
    power_profile = np.mean(spectrogram, axis=0)
    
    # Convert from dB to linear if needed
    if np.mean(power_profile) < 0:
        power_linear = 10 ** (power_profile / 20)
    else:
        power_linear = power_profile
    
    # Estimate noise floor
    noise_floor = np.percentile(power_linear, noise_percentile)
    threshold = noise_floor * 3  # 3x above noise floor
    
    # Find regions above threshold
    above_threshold = power_linear > threshold
    
    # Find start and end of each transmission
    transmissions = []
    in_transmission = False
    start_bin = 0
    
    for i in range(len(above_threshold)):
        if above_threshold[i] and not in_transmission:
            start_bin = i
            in_transmission = True
        elif not above_threshold[i] and in_transmission:
            if i - start_bin >= 10:  # Minimum 10 bins wide
                transmissions.append((start_bin, i))
            in_transmission = False
    
    # Handle case where transmission extends to end
    if in_transmission and len(above_threshold) - start_bin >= 10:
        transmissions.append((start_bin, len(above_threshold)))
    
    return transmissions


def extract_transmission(spectrogram, start_bin, end_bin, target_bins=200):
    """
    Extract a single transmission from the spectrogram.
    
    Args:
        spectrogram: 2D numpy array (time x frequency_bins)
        start_bin: start frequency bin
        end_bin: end frequency bin
        target_bins: target number of bins (2 MHz = 200 bins)
    
    Returns:
        2D numpy array of the transmission
    """
    # Extract the transmission region
    tx = spectrogram[:, start_bin:end_bin]
    
    # Crop to max 50 time segments
    tx = tx[:50, :]
    
    # If wider than target_bins, take the center portion
    if tx.shape[1] > target_bins:
        center = tx.shape[1] // 2
        start = center - target_bins // 2
        end = start + target_bins
        tx = tx[:, start:end]
    
    return tx


# ============================================================
# SECTION 4: Model Loading (TensorFlow Compatibility)
# ============================================================

def load_models():
    """
    Load pretrained AutoEncoder, LSTM classifier, and Scaler.
    Handles compatibility issues between TensorFlow versions.
    
    Returns:
        encoder, classifier, scaler
    """
    import joblib
    
    print("Loading pretrained models...")
    
    # Load scaler
    scaler = joblib.load(SCALER_PATH)
    print("  ✓ Scaler loaded")
    
    # Build AutoEncoder architecture
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    
    # Encoder: 33 -> 64 -> 32 -> 16
    encoder = Sequential()
    encoder.add(Dense(units=64, activation='relu', input_shape=[33]))
    encoder.add(Dense(units=32, activation='relu'))
    encoder.add(Dense(units=16, activation='relu'))
    
    # Decoder: 16 -> 32 -> 64 -> 33
    decoder = Sequential()
    decoder.add(Dense(units=16, activation='relu', input_shape=[16]))
    decoder.add(Dense(units=32, activation='relu'))
    decoder.add(Dense(units=64, activation='relu'))
    decoder.add(Dense(units=33))
    
    # Full autoencoder
    autoencoder = Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    
    try:
        autoencoder.load_weights(ENCODER_PATH)
        print("  ✓ AutoEncoder loaded (33 → 16 features)")
    except Exception as e:
        print(f"  ⚠ AutoEncoder weight loading warning: {e}")
        print("    Continuing with initialized weights...")
    
    # Build LSTM Classifier
    from tensorflow.keras.layers import LSTM, Dropout
    
    classifier = Sequential()
    classifier.add(LSTM(32, activation='relu', input_shape=(16, 1), return_sequences=True))
    classifier.add(LSTM(16, activation='relu'))
    classifier.add(Dense(16, activation='softmax'))
    classifier.add(Dropout(0.001))
    classifier.add(Dense(6, activation='softmax'))
    
    try:
        classifier.load_weights(CLASSIFIER_PATH)
        print("  ✓ LSTM Classifier loaded (6 classes)")
    except Exception as e:
        print(f"  ⚠ Classifier weight loading warning: {e}")
        print("    Continuing with initialized weights...")
    
    return encoder, classifier, scaler


# ============================================================
# SECTION 5: Classification Pipeline
# ============================================================

def classify_transmission(tx_data, encoder, classifier, scaler):
    """
    Classify a single transmission.
    
    Args:
        tx_data: 2D numpy array (time x frequency_bins)
        encoder: AutoEncoder model
        classifier: LSTM classifier model
        scaler: StandardScaler
    
    Returns:
        (label_name, confidence, entropy)
    """
    import pandas as pd
    
    # Extract features
    features = extract_features(tx_data)
    
    # Reshape for scaler (expects 2D array)
    features_2d = features.reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features_2d)
    
    # Encode through AutoEncoder (33 -> 16)
    features_encoded = encoder.predict(features_scaled, verbose=0)
    
    # Reshape for LSTM (batch, timesteps, features)
    features_lstm = features_encoded.reshape(1, 16, 1)
    
    # Classify
    predictions = classifier.predict(features_lstm, verbose=0)
    
    # Get predicted class and confidence
    pred_class = np.argmax(predictions[0])
    confidence = predictions[0][pred_class]
    
    # Calculate entropy (for uncertainty detection)
    probs = predictions[0] + 1e-10  # Avoid log(0)
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(probs))
    normalized_entropy = entropy / max_entropy
    
    # If entropy > 0.7, classify as unknown
    if normalized_entropy > 0.7:
        return 'unkn', confidence, normalized_entropy
    
    return LABELS[pred_class], confidence, normalized_entropy


# ============================================================
# SECTION 6: Main Processing
# ============================================================

def process_file(filepath, encoder, classifier, scaler):
    """
    Process a single .npy file and classify all transmissions.
    
    Args:
        filepath: path to .npy file
        encoder, classifier, scaler: pretrained models
    
    Returns:
        List of (frequency_range, label, confidence) tuples
    """
    # Load spectrogram
    spectrogram = np.load(filepath, allow_pickle=True)
    
    # Ensure 2D
    if spectrogram.ndim == 1:
        spectrogram = spectrogram.reshape(1, -1)
    
    # Limit to 100 time segments for speed
    spectrogram = spectrogram[:100, :]
    
    # Get filename info
    filename = os.path.basename(filepath)
    
    # Detect transmissions
    transmissions = detect_transmissions(spectrogram)
    
    results = []
    
    for start_bin, end_bin in transmissions:
        # Extract transmission
        tx_data = extract_transmission(spectrogram, start_bin, end_bin)
        
        # Skip if too small
        if tx_data.size < 10:
            continue
        
        # Classify
        label, confidence, entropy = classify_transmission(
            tx_data, encoder, classifier, scaler
        )
        
        results.append({
            'start_bin': start_bin,
            'end_bin': end_bin,
            'label': label,
            'confidence': confidence,
            'entropy': entropy
        })
    
    return results, spectrogram.shape


def main():
    """Main function to run the classification pipeline."""
    
    print("=" * 60)
    print("  Spectrum Classification Framework")
    print("  Based on: IEEE INFOCOM 2023 Paper")
    print("=" * 60)
    print()
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        print("Please ensure spectrum_bands/ folder exists in the project directory.")
        return
    
    # Load models
    encoder, classifier, scaler = load_models()
    print()
    
    # Find all .npy files
    npy_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith('.npy'):
                npy_files.append(os.path.join(root, f))
    
    print(f"Found {len(npy_files)} data files to process")
    print()
    
    # Process each file
    all_results = {}
    total_transmissions = 0
    label_counts = {}
    
    for filepath in sorted(npy_files)[:5]:  # Process first 5 files for demo
        filename = os.path.basename(filepath)
        print(f"Processing: {filename}")
        
        results, shape = process_file(filepath, encoder, classifier, scaler)
        
        print(f"  Shape: {shape}")
        print(f"  Detected {len(results)} transmissions:")
        
        for i, res in enumerate(results):
            print(f"    TX {i+1}: Bin {res['start_bin']}-{res['end_bin']} → "
                  f"{res['label'].upper()} (confidence: {res['confidence']*100:.1f}%)")
            
            # Count labels
            label = res['label']
            label_counts[label] = label_counts.get(label, 0) + 1
            total_transmissions += 1
        
        all_results[filename] = results
        print()
    
    # Summary
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"Total transmissions detected: {total_transmissions}")
    print()
    print("Technology breakdown:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = (count / total_transmissions * 100) if total_transmissions > 0 else 0
        print(f"  {label.upper():8s}: {count:3d} ({pct:.1f}%)")
    print()
    print("Done!")


if __name__ == '__main__':
    main()