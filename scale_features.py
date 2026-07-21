#!/usr/bin/env python3
"""
Scale the 33 extracted features using the pre-trained MinMaxScaler.

Input:  features_33.csv (11,400 rows × 33 features)
Output: features_33_scaled.csv (same rows, features scaled to [0, 1])

The scaler was pre-trained on the original paper's training set and saved
with joblib. We only call .transform() — never .fit().
"""

import os
import numpy as np
import pandas as pd
import joblib

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(
    BASE_DIR,
    'PSD-technology-classification-framework',
    'TCpackage', 'resources', 'scaler', '_AE16_LSTM_Scaler_.save'
)
INPUT_CSV = os.path.join(BASE_DIR, 'features_33.csv')
OUTPUT_CSV = os.path.join(BASE_DIR, 'features_33_scaled.csv')

# The 33 feature names — must match the order the scaler expects
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

# Metadata columns (not scaled)
META_COLS = ['file', 'sensor', 'technology', 'country', 'freq_start', 'freq_end', 'row']


def main():
    print("=" * 60)
    print("  Scaling 33 features with pre-trained MinMaxScaler")
    print("=" * 60)
    print()

    # Step 1: Load scaler
    print(f"Loading scaler from: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    print(f"  Type: {type(scaler).__name__}")
    print(f"  Features expected: {scaler.n_features_in_}")
    print(f"  Feature range: {scaler.feature_range}")
    print()

    # Step 2: Load CSV
    print(f"Loading features from: {INPUT_CSV}")
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
    print(f"  Any NaN before scaling: {np.isnan(X).sum()}")
    print(f"  Any Inf before scaling: {np.isinf(X).sum()}")
    print()

    # Step 5: Scale
    print("Applying scaler.transform()...")
    X_scaled = scaler.transform(X)
    print(f"  Scaled matrix shape: {X_scaled.shape}")
    print(f"  Min value: {X_scaled.min():.6f}")
    print(f"  Max value: {X_scaled.max():.6f}")
    print(f"  Any NaN after scaling: {np.isnan(X_scaled).sum()}")
    print(f"  Any Inf after scaling: {np.isinf(X_scaled).sum()}")
    print()

    # Step 6: Build output DataFrame
    scaled_df = df[META_COLS].copy()
    for i, feat_name in enumerate(FEATURE_NAMES):
        scaled_df[feat_name] = X_scaled[:, i]

    # Step 7: Save
    scaled_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved scaled features to: {OUTPUT_CSV}")
    print(f"  Rows: {len(scaled_df)}")
    print(f"  Columns: {list(scaled_df.columns)}")
    print()

    # Step 8: Preview
    print("Preview (first 3 rows, first 6 feature columns):")
    preview_cols = META_COLS[:2] + FEATURE_NAMES[:6]
    print(scaled_df[preview_cols].head(3).to_string(index=False))
    print()

    # Step 9: Stats per feature (first few)
    print("Per-feature stats (first 5 features):")
    for feat in FEATURE_NAMES[:5]:
        vals = scaled_df[feat]
        print(f"  {feat:35s}  min={vals.min():.4f}  max={vals.max():.4f}  mean={vals.mean():.4f}")

    print()
    print("DONE")


if __name__ == '__main__':
    main()
