#!/usr/bin/env python3
"""
Step 4: Feature Extraction Function
Extract 33 statistical features using tsfresh (same as author's code).
"""

import numpy as np
import pandas as pd
from tsfresh.feature_extraction.feature_calculators import (
    abs_energy,
    absolute_sum_of_changes,
    benford_correlation,
    cid_ce,
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
    number_cwt_peaks,
    number_peaks,
    quantile,
    root_mean_square,
    skewness,
    standard_deviation,
    sum_of_reoccurring_values,
    sum_values,
    variance,
    variation_coefficient
)

def extract_33_features(data, id_sns=0):
    """
    Extract 33 statistical features from PSD data.
    
    Args:
        data: 2D numpy array (time_segments, freq_bins)
        id_sns: sensor ID (default: 0)
    
    Returns:
        features: numpy array (time_segments, 33)
        columns: list of feature names
    """
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df_transf = pd.DataFrame()
    
    # Extract 33 features (same as author's extract_statitics method)
    df_transf['abs_energy'] = df.apply(lambda x_vector: abs_energy(x_vector), axis=1)
    df_transf['absolute_sum_of_changes'] = df.apply(lambda x_vector: absolute_sum_of_changes(x_vector), axis=1)
    df_transf['benford_correlation'] = df.apply(lambda x_vector: benford_correlation(x_vector), axis=1)
    df_transf['cid_ce'] = df.apply(lambda x_vector: cid_ce(x_vector, True), axis=1)
    df_transf['count_above_mean'] = df.apply(lambda x_vector: count_above_mean(x_vector), axis=1)
    df_transf['count_below_mean'] = df.apply(lambda x_vector: count_below_mean(x_vector), axis=1)
    df_transf['first_location_of_maximum'] = df.apply(lambda x_vector: first_location_of_maximum(x_vector), axis=1)
    df_transf['first_location_of_minimum'] = df.apply(lambda x_vector: first_location_of_minimum(x_vector), axis=1)
    df_transf['has_duplicate'] = df.apply(lambda x_vector: has_duplicate(x_vector), axis=1)
    df_transf['has_duplicate_max'] = df.apply(lambda x_vector: has_duplicate_max(x_vector), axis=1)
    df_transf['has_duplicate_min'] = df.apply(lambda x_vector: has_duplicate_min(x_vector), axis=1)
    df_transf['kurtosis'] = df.apply(lambda x_vector: kurtosis(x_vector), axis=1)
    df_transf['last_location_of_maximum'] = df.apply(lambda x_vector: last_location_of_maximum(x_vector), axis=1)
    df_transf['last_location_of_minimum'] = df.apply(lambda x_vector: last_location_of_minimum(x_vector), axis=1)
    df_transf['longest_strike_above_mean'] = df.apply(lambda x_vector: longest_strike_above_mean(x_vector), axis=1)
    df_transf['longest_strike_below_mean'] = df.apply(lambda x_vector: longest_strike_below_mean(x_vector), axis=1)
    df_transf['maximum'] = df.apply(lambda x_vector: maximum(x_vector), axis=1)
    df_transf['mean'] = df.apply(lambda x_vector: mean(x_vector), axis=1)
    df_transf['mean_abs_change'] = df.apply(lambda x_vector: mean_abs_change(x_vector), axis=1)
    df_transf['mean_change'] = df.apply(lambda x_vector: mean_change(x_vector), axis=1)
    df_transf['mean_second_derivative_central'] = df.apply(
        lambda x_vector: mean_second_derivative_central(x_vector), axis=1)
    df_transf['median'] = df.apply(lambda x_vector: median(x_vector), axis=1)
    df_transf['minimum'] = df.apply(lambda x_vector: minimum(x_vector), axis=1)
    df_transf['number_cwt_peaks'] = df.apply(lambda x_vector: number_cwt_peaks(x_vector, n=3), axis=1)
    df_transf['number_peaks'] = df.apply(lambda x_vector: number_peaks(x_vector, n=3), axis=1)
    df_transf['number_cwt_peaks'] = df.apply(lambda x_vector: number_cwt_peaks(x_vector, n=3), axis=1)
    df_transf['quantile'] = df.apply(lambda x_vector: quantile(x_vector, 0.5), axis=1)
    df_transf['root_mean_square'] = df.apply(lambda x_vector: root_mean_square(x_vector), axis=1)
    df_transf['skewness'] = df.apply(lambda x_vector: skewness(x_vector), axis=1)
    df_transf['standard_deviation'] = df.apply(lambda x_vector: standard_deviation(x_vector), axis=1)
    df_transf['sum_of_reoccurring_values'] = df.apply(lambda x_vector: sum_of_reoccurring_values(x_vector), axis=1)
    df_transf['sum_values'] = df.apply(lambda x_vector: sum_values(x_vector), axis=1)
    df_transf['variance'] = df.apply(lambda x_vector: variance(x_vector), axis=1)
    df_transf['variation_coefficient'] = df.apply(lambda x_vector: variation_coefficient(x_vector), axis=1)
    df_transf['Id_sensor'] = id_sns
    
    # Replace inf with NaN and fill NaN for specific columns (same as author's refine_df)
    df_transf.replace([np.inf], np.nan, inplace=True)
    df_transf['skewness'] = df_transf['skewness'].fillna(0)
    df_transf['mean_second_derivative_central'] = df_transf['mean_second_derivative_central'].fillna(0)
    df_transf['kurtosis'] = df_transf['kurtosis'].fillna(0)
    
    # Get features (remove Id_sensor column)
    features = df_transf.values[:, :-1]  # Remove last column (Id_sensor)
    columns = list(df_transf.columns.values)[:-1]  # Remove Id_sensor from column names
    
    return features, columns

def test_feature_extraction():
    """Test feature extraction on sample data."""
    print("Testing feature extraction...")
    
    # Load sample FM file
    sample_file = 'dataset/opt/shared/alessio/Data_Transmitters_Identification/spectrum_bands_2/miguel_murcia/Dec_1/SpectrumBands_85_105_fm_Esp_85_105.npy'
    data = np.load(sample_file)
    print(f"Loaded data shape: {data.shape}")
    
    # Truncate to 50 time segments
    data = data[:50, :]
    print(f"Truncated data shape: {data.shape}")
    
    # Cast to float32
    data = data.astype(np.float32)
    print(f"Data dtype: {data.dtype}")
    
    # Extract features
    features, columns = extract_33_features(data)
    print(f"Features shape: {features.shape}")
    print(f"Number of features: {len(columns)}")
    print(f"Feature names: {columns[:5]}... (showing first 5)")
    
    # Verify shape
    assert features.shape == (50, 33), f"Expected (50, 33), got {features.shape}"
    assert len(columns) == 33, f"Expected 33 features, got {len(columns)}"
    
    print("\nFeature extraction test passed!")
    return features, columns

if __name__ == "__main__":
    features, columns = test_feature_extraction()
