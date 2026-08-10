import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt
from scipy.stats import skew, kurtosis


class FeatureExtractor:
    """
    Feature extraction module implementing 33 statistical features
    as described in Section IV-A of the paper.
    
    Each PSD segment is transformed into a fixed-length feature vector.
    """

    def __init__(self):
        self.feature_names = [
            'abs_energy', 'absolute_sum_of_changes', 'benford_correlation',
            'cid_ce', 'count_above_mean', 'count_below_mean',
            'first_location_of_maximum', 'first_location_of_minimum',
            'has_duplicate', 'has_duplicate_max', 'has_duplicate_min',
            'kurtosis', 'last_location_of_maximum', 'last_location_of_minimum',
            'longest_strike_above_mean', 'longest_strike_below_mean',
            'maximum', 'mean', 'mean_abs_change', 'mean_change',
            'mean_second_derivative_central', 'median', 'minimum',
            'number_cwt_peaks', 'number_peaks',
            'root_mean_square', 'skewness', 'standard_deviation',
            'sum_of_reoccurring_values', 'sum_values', 'variance',
            'variation_coefficient'
        ]

    def extract_features(self, psd_segment):
        """
        Extract 32 features from a single PSD segment.
        
        Args:
            psd_segment: 1D numpy array of PSD values (one row of the spectrogram)
        
        Returns:
            1D numpy array of 32 features
        """
        x = np.array(psd_segment, dtype=float)
        n = len(x)

        features = []

        # 0. abs_energy: sum of x^2
        features.append(np.sum(x ** 2))

        # 1. absolute_sum_of_changes: sum of |diff(x)|
        features.append(np.sum(np.abs(np.diff(x))))

        # 2. benford_correlation: correlation with Benford's law
        features.append(self._benford_correlation(x))

        # 3. cid_ce: complexity estimate (sqrt of sum of squared differences)
        features.append(self._cid_ce(x))

        # 4. count_above_mean: count of values above mean
        features.append(np.sum(x > np.mean(x)))

        # 5. count_below_mean: count of values below mean
        features.append(np.sum(x < np.mean(x)))

        # 6. first_location_of_maximum: argmax / len
        features.append(np.argmax(x) / n if n > 0 else 0)

        # 7. first_location_of_minimum: argmin / len
        features.append(np.argmin(x) / n if n > 0 else 0)

        # 8. has_duplicate: whether there are duplicate values
        features.append(1 if len(np.unique(x)) < n else 0)

        # 9. has_duplicate_max: whether max appears more than once
        features.append(1 if np.sum(x == np.max(x)) > 1 else 0)

        # 10. has_duplicate_min: whether min appears more than once
        features.append(1 if np.sum(x == np.min(x)) > 1 else 0)

        # 11. kurtosis: kurtosis of the distribution
        features.append(float(kurtosis(x, fisher=True)) if n > 1 else 0)

        # 12. last_location_of_maximum: (len - 1 - argmax) / len
        features.append((n - 1 - np.argmax(x)) / n if n > 0 else 0)

        # 13. last_location_of_minimum: (len - 1 - argmin) / len
        features.append((n - 1 - np.argmin(x)) / n if n > 0 else 0)

        # 14. longest_strike_above_mean
        features.append(self._longest_strike(x, above=True))

        # 15. longest_strike_below_mean
        features.append(self._longest_strike(x, above=False))

        # 16. maximum
        features.append(np.max(x))

        # 17. mean
        features.append(np.mean(x))

        # 18. mean_abs_change: mean of |diff(x)|
        features.append(np.mean(np.abs(np.diff(x))) if n > 1 else 0)

        # 19. mean_change: mean of diff(x)
        features.append(np.mean(np.diff(x)) if n > 1 else 0)

        # 20. mean_second_derivative_central
        features.append(self._mean_second_derivative_central(x))

        # 21. median
        features.append(np.median(x))

        # 22. minimum
        features.append(np.min(x))

        # 23. number_cwt_peaks: CWT-based peak count
        features.append(self._number_cwt_peaks(x))

        # 24. number_peaks: peak count using find_peaks
        features.append(self._number_peaks(x))

        # 25. root_mean_square
        features.append(np.sqrt(np.mean(x ** 2)))

        # 26. skewness
        features.append(float(skew(x)) if n > 1 else 0)

        # 27. standard_deviation
        features.append(np.std(x))

        # 28. sum_of_reoccurring_values: sum of values that appear more than once
        features.append(self._sum_of_reoccurring_values(x))

        # 29. sum_values
        features.append(np.sum(x))

        # 30. variance
        features.append(np.var(x))

        # 31. variation_coefficient: std / mean
        mean_val = np.mean(x)
        features.append(np.std(x) / mean_val if mean_val != 0 else 0)

        return np.array(features, dtype=float)

    def _benford_correlation(self, x):
        """Compute Benford's law correlation."""
        # Get first significant digit of each value
        abs_x = np.abs(x[(x != 0)])
        if len(abs_x) == 0:
            return 0

        # Get first digit
        first_digits = []
        for val in abs_x:
            s = f"{val:.10f}".lstrip('0').lstrip('.')
            if s and s[0].isdigit() and s[0] != '0':
                first_digits.append(int(s[0]))

        if len(first_digits) == 0:
            return 0

        # Benford's expected distribution
        benford = np.array([np.log10(1 + 1/d) for d in range(1, 10)])

        # Observed distribution
        observed = np.zeros(9)
        for d in first_digits:
            if 1 <= d <= 9:
                observed[d - 1] += 1

        # Normalize
        if np.sum(observed) > 0:
            observed = observed / np.sum(observed)
            # Pearson correlation
            correlation = np.corrcoef(benford, observed)[0, 1]
            return correlation if not np.isnan(correlation) else 0
        return 0

    def _cid_ce(self, x, normalize=True):
        """CID_CE: Complexity estimate."""
        if normalize:
            std = np.std(x)
            if std > 0:
                x_norm = (x - np.mean(x)) / std
            else:
                x_norm = x - np.mean(x)
        else:
            x_norm = x
        return np.sqrt(np.sum(np.diff(x_norm) ** 2))

    def _longest_strike(self, x, above=True):
        """Compute longest consecutive strike above or below mean."""
        mean_val = np.mean(x)
        if above:
            mask = x > mean_val
        else:
            mask = x < mean_val

        longest = 0
        current = 0
        for val in mask:
            if val:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest

    def _mean_second_derivative_central(self, x):
        """Mean of second central derivative."""
        n = len(x)
        if n < 3:
            return 0
        # Central second derivative: (x[i+2] - 2*x[i+1] + x[i]) / 2
        second_deriv = (x[2:] - 2 * x[1:-1] + x[:-2]) / 2
        return np.mean(second_deriv)

    def _number_cwt_peaks(self, x, n=3):
        """Number of CWT peaks using scipy find_peaks_cwt."""
        try:
            # find_peaks_cwt requires 1D signal
            peaks = find_peaks_cwt(x, np.arange(1, n + 1))
            return len(peaks)
        except Exception:
            return 0

    def _number_peaks(self, x, n=3):
        """Number of peaks using scipy find_peaks with distance=n, prominence=n."""
        try:
            peaks, _ = find_peaks(x, distance=n, prominence=n)
            return len(peaks)
        except Exception:
            return 0

    def _sum_of_reoccurring_values(self, x):
        """Sum of values that appear more than once."""
        unique, counts = np.unique(x, return_counts=True)
        reoccurring = unique[counts > 1]
        return np.sum(reoccurring)

    def extract_features_from_matrix(self, psd_matrix, strategy='hopping'):
        """
        Extract features from a PSD matrix using hopping strategy.
        
        From the paper (Section VII-B.3):
        "The hopping strategy extracts features in chunks of 2 MHz 
        from the first frequency bin."
        
        Args:
            psd_matrix: 2D numpy array (K, M) - K time sweeps, M frequency bins
            strategy: 'hopping' to use paper's hopping strategy
        
        Returns:
            1D numpy array of concatenated features
        """
        if strategy == 'hopping':
            return self._hopping_strategy(psd_matrix)
        else:
            # Use entire segment
            return self.extract_features(psd_matrix.flatten())

    def _hopping_strategy(self, psd_matrix):
        """
        Hopping strategy: extract features in chunks of 2 MHz (215 bins)
        from the first frequency bin.
        
        From the paper:
        "The hopping strategy extracts features in chunks of 2 MHz 
        from the first frequency bin."
        
        This means we take chunks of 215 bins starting from bin 0,
        and extract features from each chunk.
        """
        chunk_size = 215
        n_bins = psd_matrix.shape[1]

        # Get the first time sweep (single PSD measurement)
        # The paper uses "only one single PSD measurement" for classification
        psd_row = psd_matrix[0, :]  # shape: (M,)

        all_features = []

        # Extract features in chunks of 215 bins
        num_chunks = len(psd_row) // chunk_size
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = psd_row[start:end]
            features = self.extract_features(chunk)
            all_features.extend(features)

        # If there are remaining bins, also extract features
        remainder = len(psd_row) % chunk_size
        if remainder > 0:
            chunk = psd_row[num_chunks * chunk_size:]
            features = self.extract_features(chunk)
            all_features.extend(features)

        return np.array(all_features)


if __name__ == "__main__":
    # Test feature extraction
    extractor = FeatureExtractor()

    # Create a test PSD segment
    test_segment = np.random.randn(215) * 10 - 50

    features = extractor.extract_features(test_segment)
    print(f"Number of features: {len(features)}")
    print(f"Feature names: {extractor.feature_names}")
    print(f"Feature values: {features}")
