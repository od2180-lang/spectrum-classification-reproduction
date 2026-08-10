import numpy as np
from scipy.signal import find_peaks


class TransmissionDetectionSystem:
    """
    Transmission Detection System (TDS) as described in Section III of the paper.
    
    Three main blocks:
    1. Noise level computation
    2. Energy detector
    3. Peaks finding and edge detection
    """

    def __init__(self, noise_threshold_db=5, cv_threshold=1.0, peak_threshold_db=3):
        """
        Parameters from the paper:
        - noise_threshold_db: 5 dB threshold above noise for energy detector (ITU standard)
        - cv_threshold: CV=1 threshold to detect sub-transmissions
        - peak_threshold_db: 3 dB below peak for edge detection
        """
        self.noise_threshold_db = noise_threshold_db
        self.cv_threshold = cv_threshold
        self.peak_threshold_db = peak_threshold_db

    def compute_noise_level(self, spectrum_data):
        """
        Block 1: Noise level computation (Section III)
        
        From the paper:
        1. Convert PSD from dB to amplitude: Ym = 10^(Xm/20)
        2. Average bins for every single PSD segment of length M=215
        3. Find which portion (PSD segment) of the full scanned spectrum has minimum value
           j = arg{μj = min(μm)}; m = 1, 2, ..., 870
        4. Compute standard deviation σj on the selected portion
        5. Apply 3-sigma rule: η = μj + 3σj
        
        The paper works on full spectrum (870 freq segments × 215 bins each).
        For our filtered-band data, we adapt:
        - Convert to amplitude
        - Average over time to get a single spectrum
        - Divide into segments of 215 bins
        - Find the segment with minimum average
        - Use that segment's bins to compute μj and σj
        """
        # spectrum_data shape: (K, M) where K = time segments, M = frequency bins
        
        # Step 1: Convert from dB to amplitude
        amplitude_data = np.power(10, spectrum_data / 20)
        
        # Step 2: Average over time to get a single spectrum
        averaged_spectrum = np.mean(amplitude_data, axis=0)  # shape: (M,)
        
        # Step 3: Divide into segments of 215 bins and find the quietest
        segment_size = 215
        num_segments = len(averaged_spectrum) // segment_size
        
        if num_segments == 0:
            # Spectrum smaller than 215 bins, use entire spectrum
            mu_j = np.mean(averaged_spectrum)
            sigma_j = np.std(averaged_spectrum)
        else:
            # Compute mean for each segment
            segment_means = []
            for m in range(num_segments):
                start = m * segment_size
                end = start + segment_size
                seg_mean = np.mean(averaged_spectrum[start:end])
                segment_means.append(seg_mean)
            
            # Find segment with minimum mean
            j = np.argmin(segment_means)
            
            # Compute μj and σj using ALL bins in segment j (across all time sweeps)
            start = j * segment_size
            end = start + segment_size
            noise_bins = amplitude_data[:, start:end]  # shape: (K, 215)
            mu_j = np.mean(noise_bins)
            sigma_j = np.std(noise_bins)
        
        # Step 4: Apply 3-sigma rule: η = μj + 3σj
        noise_level_linear = mu_j + 3 * sigma_j
        
        # Convert back to dB
        noise_level_db = 20 * np.log10(noise_level_linear)
        
        return noise_level_db

    def energy_detector(self, spectrum_data, noise_level_db):
        """
        Block 2: Energy detector (Section III)
        
        Steps:
        1. Operate on K consecutive time segments
        2. Compute average over time per bin
        3. Detect active transmission when AVG - η ≥ 5 dB
        4. Declare end when AVG - η < threshold
        """
        # Convert to amplitude
        amplitude_data = np.power(10, spectrum_data / 20)

        # Compute average over time per bin
        avg_spectrum = np.mean(amplitude_data, axis=0)

        # Convert average to dB
        avg_spectrum_db = 20 * np.log10(avg_spectrum)

        # Compute difference from noise level
        diff_db = avg_spectrum_db - noise_level_db

        # Detect active bins
        active_bins = diff_db >= self.noise_threshold_db

        # Find contiguous regions (transmissions)
        transmissions = []
        in_transmission = False
        start_idx = 0

        for i in range(len(active_bins)):
            if active_bins[i] and not in_transmission:
                # Start of new transmission
                start_idx = i
                in_transmission = True
            elif not active_bins[i] and in_transmission:
                # End of transmission
                end_idx = i - 1
                transmissions.append((start_idx, end_idx))
                in_transmission = False

        # Handle transmission that extends to end of spectrum
        if in_transmission:
            end_idx = len(active_bins) - 1
            transmissions.append((start_idx, end_idx))

        return transmissions

    def find_peaks_and_edges(self, transmission_data, start_idx):
        """
        Block 3: Peaks finding and edge detection (Section III)
        
        Steps:
        1. Compute Coefficient of Variation (CV) = σX/μX
        2. If CV < 1: transmission is stable, add to list
        3. If CV ≥ 1: find sub-transmissions using peaks and edge detection
        4. Edge detection: 3 dB threshold below peak
        """
        # transmission_data shape: (K, M) where M is the transmission bandwidth
        # Compute CV
        mean_val = np.mean(transmission_data)
        std_val = np.std(transmission_data)

        if mean_val == 0:
            return [(start_idx, start_idx + transmission_data.shape[1] - 1)]

        cv = std_val / mean_val

        if cv < self.cv_threshold:
            # Transmission is stable
            return [(start_idx, start_idx + transmission_data.shape[1] - 1)]
        else:
            # Need to find sub-transmissions
            # Average over time
            averaged = np.mean(np.power(10, transmission_data / 20), axis=0)

            # Find peaks
            prominence = np.mean(averaged) + np.std(averaged)
            peaks, properties = find_peaks(averaged, distance=10, width=4, prominence=prominence)

            if len(peaks) == 0:
                return [(start_idx, start_idx + transmission_data.shape[1] - 1)]

            # For each peak, find edges using 3 dB threshold
            sub_transmissions = []
            for peak in peaks:
                peak_value = averaged[peak]

                # Search left for edge
                left_edge = 0
                for i in range(peak, -1, -1):
                    if 20 * np.log10(peak_value / averaged[i]) > self.peak_threshold_db:
                        left_edge = i
                        break

                # Search right for edge
                right_edge = len(averaged) - 1
                for i in range(peak, len(averaged)):
                    if 20 * np.log10(peak_value / averaged[i]) > self.peak_threshold_db:
                        right_edge = i
                        break

                # Add sub-transmission
                sub_transmissions.append((start_idx + left_edge, start_idx + right_edge))

            return sub_transmissions

    def detect_transmissions(self, spectrum_data):
        """
        Main TDS pipeline.
        
        Input: spectrum_data - numpy array of shape (K, M)
               where K = time segments, M = frequency bins
               Values are in dB scale
        
        Output: list of (start_bin, end_bin) tuples for each detected transmission
        """
        # Block 1: Compute noise level
        noise_level_db = self.compute_noise_level(spectrum_data)

        # Block 2: Energy detector
        coarse_transmissions = self.energy_detector(spectrum_data, noise_level_db)

        # Block 3: Find peaks and edges for each detected transmission
        final_transmissions = []
        for start, end in coarse_transmissions:
            # Extract transmission data
            tx_data = spectrum_data[:, start:end + 1]

            # Find peaks and edges
            sub_txs = self.find_peaks_and_edges(tx_data, start)
            final_transmissions.extend(sub_txs)

        return final_transmissions


def load_and_detect(file_path):
    """
    Load a .npy file and run TDS to detect transmissions.
    
    Returns: list of (start_bin, end_bin) tuples
    """
    data = np.load(file_path, allow_pickle=True)
    tds = TransmissionDetectionSystem()
    transmissions = tds.detect_transmissions(data)
    return transmissions


if __name__ == "__main__":
    # Test with a sample file
    import os

    dataset_path = "/home/jovyan/work/project/dataset/opt/shared/alessio/Data_Transmitters_Identification/spectrum_bands_2"

    # Find a sample file
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if f.endswith('.npy'):
                file_path = os.path.join(root, f)
                print(f"\nProcessing: {f}")

                data = np.load(file_path, allow_pickle=True)
                print(f"Data shape: {data.shape}")

                tds = TransmissionDetectionSystem()
                transmissions = tds.detect_transmissions(data)
                print(f"Detected {len(transmissions)} transmission(s):")
                for i, (start, end) in enumerate(transmissions):
                    print(f"  TX {i+1}: bins [{start}, {end}] (width: {end - start + 1})")

                break
        break
