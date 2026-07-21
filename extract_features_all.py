#!/usr/bin/env python3
"""
Extract 33 tsfresh features from all .npy spectrogram files.

This script replicates the feature extraction from the paper:
"A Framework for Wireless Technology Classification using Crowdsensing Platforms"

What it does, step by step:
1. Finds every .npy file in the spectrum_bands/ folder
2. For each file:
   a. Loads the file (a spreadsheet-like grid of numbers)
   b. Keeps only the first 50 rows (time segments)
   c. Keeps only the middle 200 columns (frequency bins, ~2 MHz)
   d. For each of the 50 rows, calculates 33 math statistics
   e. Replaces any broken values (infinity, NaN) with 0
3. Saves everything to a CSV file
"""

import os
import re
import numpy as np
import pandas as pd
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

# ============================================================
# CONFIGURATION - These are the settings you can change
# ============================================================

# Where to find the .npy data files
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spectrum_bands')

# Where to save the output CSV
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'features_33.csv')

# How many rows (time segments) to keep from each file
# The original paper's code uses 50
MAX_TIME_SEGMENTS = 50

# How many columns (frequency bins) to keep, centered in the middle
# 200 bins ≈ 2 MHz of bandwidth
TARGET_FREQ_BINS = 200

# The 33 math statistics we calculate for each row
# These are called "tsfresh features" - they describe the shape of the data
FEATURE_NAMES = [
    'abs_energy',               # Total energy of the signal
    'absolute_sum_of_changes',  # How much the signal jumps around
    'benford_correlation',       # Pattern in leading digits
    'cid_ce',                   # Complexity measure
    'count_above_mean',         # How many points are above the average
    'count_below_mean',         # How many points are below the average
    'first_location_of_maximum',# Where the highest point is (as a fraction)
    'first_location_of_minimum',# Where the lowest point is (as a fraction)
    'has_duplicate',            # Are there repeated values?
    'has_duplicate_max',        # Does the maximum value appear more than once?
    'has_duplicate_min',        # Does the minimum value appear more than once?
    'kurtosis',                 # How "peaked" is the distribution
    'last_location_of_maximum', # Where the last highest point is
    'last_location_of_minimum', # Where the last lowest point is
    'longest_strike_above_mean',# Longest run of points above average
    'longest_strike_below_mean',# Longest run of points below average
    'maximum',                  # Highest value
    'mean',                     # Average value
    'mean_abs_change',          # Average jump between consecutive points
    'mean_change',              # Average change (can be negative)
    'mean_second_derivative_central',# Acceleration of the signal
    'median',                   # Middle value
    'minimum',                  # Lowest value
    'number_cwt_peaks',         # How many peaks (smoothed detection)
    'number_peaks',             # How many peaks (sharp detection)
    'quantile',                 # Median (50th percentile)
    'root_mean_square',         # RMS - like average but emphasizing big values
    'skewness',                 # Is the data skewed left or right?
    'standard_deviation',       # How spread out is the data
    'sum_of_reoccurring_values',# Sum of values that appear more than once
    'sum_values',               # Total sum of all values
    'variance',                 # Average squared distance from mean
    'variation_coefficient',    # Standard deviation / mean (relative spread)
]

# These are the same 33 features, but as callable functions
# Each function takes a 1D array (one row of data) and returns one number
FEATURE_FUNCTIONS = [
    abs_energy,
    absolute_sum_of_changes,
    benford_correlation,
    lambda x: cid_ce(x, True),  # normalize=True (matches article)
    count_above_mean,
    count_below_mean,
    first_location_of_maximum,
    first_location_of_minimum,
    has_duplicate,
    has_duplicate_max,
    has_duplicate_min,
    kurtosis,
    last_location_of_maximum,
    last_location_of_minimum,
    longest_strike_above_mean,
    longest_strike_below_mean,
    maximum,
    mean,
    mean_abs_change,
    mean_change,
    mean_second_derivative_central,
    median,
    minimum,
    lambda x: number_cwt_peaks(x, n=3),   # n=3 means look for peaks wider than 3 points
    lambda x: number_peaks(x, n=3),         # n=3 means peaks must be at least 3 points wide
    lambda x: quantile(x, 0.5),             # 0.5 = median (50th percentile)
    root_mean_square,
    skewness,
    standard_deviation,
    sum_of_reoccurring_values,
    sum_values,
    variance,
    variation_coefficient,
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_filename(filepath):
    """
    Pull out useful info from the filename and folder path.
    
    Example filename: SpectrumBands_85_120_fm_Esp_85_120.npy
    Example folder:   spectrum_bands/Andrew_GVA/Sep_1/
    
    Returns: (sensor, technology, country, freq_start, freq_end)
    """
    # Get the sensor name from the parent folder
    # e.g., "spectrum_bands/Andrew_GVA/Sep_1/SpectrumBands_..." -> "Andrew_GVA"
    sensor = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
    
    # Parse the filename
    filename = os.path.basename(filepath).replace('.npy', '')
    parts = filename.split('_')
    
    # Known technologies in the dataset
    KNOWN_TECHS = {'dab', 'dvbt', 'fm', 'gsm', 'lte', 'tetra'}
    
    # Find technology by looking for known tech names in the parts
    technology = ''
    tech_idx = -1
    for i, part in enumerate(parts):
        if part.lower() in KNOWN_TECHS:
            technology = part.lower()
            tech_idx = i
            break
    
    # For standard format: SpectrumBands_start_end_tech_country_start_end
    # tech is at index 3, country at index 4, freq_start at index 1, freq_end at index 2
    if tech_idx == 3:
        freq_start = parts[1] if len(parts) > 1 else ''
        freq_end = parts[2] if len(parts) > 2 else ''
        country = parts[4] if len(parts) > 4 else ''
    else:
        # For non-standard format (e.g., alcorcon files), 
        # try to find freq numbers near the technology
        freq_start = parts[tech_idx - 2] if tech_idx >= 2 else ''
        freq_end = parts[tech_idx - 1] if tech_idx >= 1 else ''
        country = parts[tech_idx + 1] if tech_idx + 1 < len(parts) else ''
    
    return sensor, technology, country, freq_start, freq_end


def extract_features_from_row(row):
    """
    Calculate 33 statistics from one row of data.
    
    A "row" is one time segment - a list of numbers representing
    signal strength at different frequencies at one moment in time.
    
    Input:  1D numpy array (e.g., 200 numbers)
    Output: list of 33 numbers (the features)
    """
    features = []
    
    for func in FEATURE_FUNCTIONS:
        try:
            val = func(row)
            # If the result is infinity or NaN, replace with 0
            if np.isnan(val) or np.isinf(val):
                val = 0.0
            features.append(float(val))
        except Exception:
            # If the function fails for any reason, use 0
            features.append(0.0)
    
    return features


def process_one_file(filepath):
    """
    Process a single .npy file and return 50 rows of 33 features each.
    
    Steps:
    1. Load the .npy file (a grid of numbers)
    2. Trim to first 50 rows (time segments)
    3. Crop to middle 200 columns (frequency bins)
    4. Extract 33 features from each of the 50 rows
    
    Input:  path to a .npy file
    Output: numpy array of shape (50, 33) or None if something went wrong
    """
    # Step 1: Load the file
    # .npy files contain numpy arrays - think of them as spreadsheets
    data = np.load(filepath, allow_pickle=True)
    
    # If the data is 1D (only one row), reshape it to 2D
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    # Step 2: Trim to first 50 rows (time segments)
    # The paper's code keeps only the first 50 time segments
    data = data[:MAX_TIME_SEGMENTS, :]
    
    # Step 3: Crop to middle 200 columns (frequency bins)
    # Find the center column, then take 100 columns on each side
    center = data.shape[1] // 2
    half = TARGET_FREQ_BINS // 2  # 100
    
    # Make sure we have enough columns to crop
    if data.shape[1] < TARGET_FREQ_BINS:
        # Not enough columns - skip this file
        print(f"    SKIPPED: only {data.shape[1]} freq bins, need {TARGET_FREQ_BINS}")
        return None
    
    data = data[:, (center - half):(center + half)]
    
    # Step 4: Extract 33 features from each row
    # We end up with a grid of shape (number_of_rows, 33)
    all_features = []
    for row_idx in range(data.shape[0]):
        row = data[row_idx, :]
        features = extract_features_from_row(row)
        all_features.append(features)
    
    return np.array(all_features, dtype=np.float32)


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    print("=" * 60)
    print("  Extracting 33 tsfresh features from all .npy files")
    print("=" * 60)
    print()
    
    # Step 1: Find all .npy files
    # os.walk goes through every folder and subfolder
    npy_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith('.npy'):
                npy_files.append(os.path.join(root, f))
    
    npy_files.sort()  # Process in alphabetical order
    print(f"Found {len(npy_files)} .npy files")
    print()
    
    # Step 2: Process each file
    all_rows = []       # Will hold all the feature data
    skipped = 0         # Count of files we skipped
    
    for file_idx, filepath in enumerate(npy_files):
        filename = os.path.basename(filepath)
        
        # Show progress every 20 files
        if file_idx % 20 == 0 or file_idx == len(npy_files) - 1:
            print(f"Processing file {file_idx + 1}/{len(npy_files)}: {filename}")
        
        # Parse metadata from filename
        sensor, technology, country, freq_start, freq_end = parse_filename(filepath)
        
        # Extract features
        features = process_one_file(filepath)
        
        if features is None:
            skipped += 1
            continue
        
        # features has shape (50, 33) - 50 rows, 33 features each
        # Now we build a table with metadata + features
        for row_idx in range(features.shape[0]):
            row_data = {
                'file': filename,
                'sensor': sensor,
                'technology': technology,
                'country': country,
                'freq_start': freq_start,
                'freq_end': freq_end,
                'row': row_idx,  # Which time segment (0-49)
            }
            # Add each of the 33 features
            for feat_idx, feat_name in enumerate(FEATURE_NAMES):
                row_data[feat_name] = features[row_idx, feat_idx]
            
            all_rows.append(row_data)
    
    print()
    
    # Step 3: Save to CSV
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    
    # Step 4: Print summary
    print("=" * 60)
    print("  DONE!")
    print("=" * 60)
    print(f"Total files processed: {len(npy_files) - skipped}")
    print(f"Files skipped (too narrow): {skipped}")
    print(f"Total rows in output: {len(df)}")
    print(f"  (should be ~{min(len(npy_files), (len(npy_files) - skipped)) * MAX_TIME_SEGMENTS})")
    print(f"Output saved to: {OUTPUT_CSV}")
    print()
    
    # Show a preview of the data
    print("Preview of first 5 rows:")
    print(df.head().to_string())
    print()
    
    # Check for bad values
    nan_count = df[FEATURE_NAMES].isna().sum().sum()
    inf_count = np.isinf(df[FEATURE_NAMES].values).sum()
    print(f"Bad values check: {nan_count} NaN, {inf_count} Inf")


if __name__ == '__main__':
    main()
