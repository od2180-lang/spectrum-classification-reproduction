#!/usr/bin/env python3
"""
Encode the 33 scaled features into 16 compressed features using the pre-trained AutoEncoder.

Input:  features_33_scaled.csv (11,400 rows × 33 scaled features)
Output: features_16.csv (same rows, 16 encoded features)

The autoencoder was pre-trained on the original paper's training set.
Architecture: 33 → 64 → 32 → 16 (encoder), 16 → 32 → 64 → 33 (decoder).
We extract only the encoder half and call .predict() — never .fit().
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODER_WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    'PSD-technology-classification-framework',
    'TCpackage', 'resources', 'save-DL-models', 'Autoencoder_DNN',
    'TrainAllSensorHop_16_feat_mse_relu', 'saved-model-49-0.0002.hdf5'
)
INPUT_CSV = os.path.join(BASE_DIR, 'features_33_scaled.csv')
OUTPUT_CSV = os.path.join(BASE_DIR, 'features_16.csv')

# The 33 feature names — must match the order the encoder expects
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

# Metadata columns (not encoded)
META_COLS = ['file', 'sensor', 'technology', 'country', 'freq_start', 'freq_end', 'row']


def build_encoder(weights_path):
    """Build the autoencoder, load weights, and extract the encoder half."""
    activ_f = 'relu'

    # Encoder: 33 → 64 → 32 → 16
    encoder = Sequential()
    encoder.add(Dense(units=64, activation=activ_f, input_shape=[33]))
    encoder.add(Dense(units=32, activation=activ_f))
    encoder.add(Dense(units=16, activation=activ_f))

    # Decoder: 16 → 32 → 64 → 33
    decoder = Sequential()
    decoder.add(Dense(units=16, activation=activ_f, input_shape=[16]))
    decoder.add(Dense(units=32, activation=activ_f))
    decoder.add(Dense(units=64, activation=activ_f))
    decoder.add(Dense(units=33))

    # Full autoencoder, load weights, then extract encoder only
    autoencoder = Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.load_weights(weights_path)

    encoder = autoencoder.layers[0]
    return encoder


def main():
    print("=" * 60)
    print("  Encoding 33 features → 16 with pre-trained AutoEncoder")
    print("=" * 60)
    print()

    # Step 1: Build encoder and load weights
    print(f"Loading encoder weights from: {ENCODER_WEIGHTS_PATH}")
    encoder = build_encoder(ENCODER_WEIGHTS_PATH)
    encoder.summary()
    print()

    # Step 2: Load CSV
    print(f"Loading scaled features from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print()

    # Step 3: Verify feature columns match
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    extra = [f for f in df.columns if f not in FEATURE_NAMES and f not in META_COLS]
    if missing:
        print(f"  ERROR: Missing feature columns: {missing}")
        return
    if extra:
        print(f"  WARNING: Extra columns found (will be ignored): {extra}")
    print(f"  All {len(FEATURE_NAMES)} feature columns present")
    print()

    # Step 4: Extract feature matrix
    X = df[FEATURE_NAMES].values.astype(np.float64)
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Any NaN before encoding: {np.isnan(X).sum()}")
    print(f"  Any Inf before encoding: {np.isinf(X).sum()}")
    print()

    # Step 5: Encode
    print("Applying encoder.predict()...")
    X_encoded = encoder.predict(X)
    print(f"  Encoded matrix shape: {X_encoded.shape}")
    print(f"  Min value: {X_encoded.min():.6f}")
    print(f"  Max value: {X_encoded.max():.6f}")
    print(f"  Any NaN after encoding: {np.isnan(X_encoded).sum()}")
    print(f"  Any Inf after encoding: {np.isinf(X_encoded).sum()}")
    print()

    # Step 6: Build output DataFrame
    encoded_df = df[META_COLS].copy()
    for i in range(16):
        encoded_df[f'enc_{i}'] = X_encoded[:, i]

    # Step 7: Save
    encoded_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved encoded features to: {OUTPUT_CSV}")
    print(f"  Rows: {len(encoded_df)}")
    print(f"  Columns: {list(encoded_df.columns)}")
    print()

    # Step 8: Preview
    enc_cols = [f'enc_{i}' for i in range(16)]
    print("Preview (first 3 rows, first 6 encoded features):")
    preview_cols = META_COLS[:2] + enc_cols[:6]
    print(encoded_df[preview_cols].head(3).to_string(index=False))
    print()

    # Step 9: Stats per encoded feature (first few)
    print("Per-feature stats (first 5 encoded features):")
    for feat in enc_cols[:5]:
        vals = encoded_df[feat]
        print(f"  {feat:10s}  min={vals.min():.4f}  max={vals.max():.4f}  mean={vals.mean():.4f}")

    print()
    print("DONE")


if __name__ == '__main__':
    main()
