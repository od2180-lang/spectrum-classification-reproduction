#!/usr/bin/env python3
"""
Step 1: 2 MHz Chunk Extraction Function
Extract center 200 bins (2 MHz) from full band data.
"""

import numpy as np

def extract_2mhz(data, target_bins=200):
    """
    Extract center 200 bins (2 MHz) from full band data.
    
    This matches the author's extract2MHz() method in TechClass.py line 170.
    The author extracts center 200 bins (not left edge as stated in paper).
    
    Parameters
    ----------
    data : numpy.ndarray
        2D array of shape (time_segments, freq_bins)
    target_bins : int
        Number of bins to extract (default: 200 for 2 MHz)
    
    Returns
    -------
    numpy.ndarray
        2D array of shape (time_segments, target_bins)
    """
    freq_bins = data.shape[1]
    
    if freq_bins <= target_bins:
        # Band is already smaller than or equal to target, use as-is
        return data
    
    # Calculate center and extract
    center = freq_bins // 2
    start = center - (target_bins // 2)
    end = start + target_bins
    
    return data[:, start:end]


def test_extract_2mhz():
    """Test the 2 MHz extraction function."""
    print("Testing extract_2mhz function...")
    
    # Test case 1: FM data (2150 bins → 200 bins)
    fm_data = np.random.randn(50, 2150).astype(np.float32)
    fm_2mhz = extract_2mhz(fm_data)
    assert fm_2mhz.shape == (50, 200), f"Expected (50, 200), got {fm_2mhz.shape}"
    print(f"✓ FM: {fm_data.shape} → {fm_2mhz.shape}")
    
    # Test case 2: TETRA data (538 bins → 200 bins)
    tetra_data = np.random.randn(50, 538).astype(np.float32)
    tetra_2mhz = extract_2mhz(tetra_data)
    assert tetra_2mhz.shape == (50, 200), f"Expected (50, 200), got {tetra_2mhz.shape}"
    print(f"✓ TETRA: {tetra_data.shape} → {tetra_2mhz.shape}")
    
    # Test case 3: Small band (74 bins → 74 bins, no change)
    small_data = np.random.randn(50, 74).astype(np.float32)
    small_2mhz = extract_2mhz(small_data)
    assert small_2mhz.shape == (50, 74), f"Expected (50, 74), got {small_2mhz.shape}"
    print(f"✓ Small: {small_data.shape} → {small_2mhz.shape}")
    
    # Test case 4: Exact 200 bins (no change)
    exact_data = np.random.randn(50, 200).astype(np.float32)
    exact_2mhz = extract_2mhz(exact_data)
    assert exact_2mhz.shape == (50, 200), f"Expected (50, 200), got {exact_2mhz.shape}"
    print(f"✓ Exact: {exact_data.shape} → {exact_2mhz.shape}")
    
    # Test case 5: Verify center extraction is correct
    # Create data with known pattern
    test_data = np.zeros((1, 100), dtype=np.float32)
    test_data[0, 50] = 1.0  # Center bin
    result = extract_2mhz(test_data, target_bins=20)
    center_idx = result.shape[1] // 2
    assert result[0, center_idx] == 1.0, "Center extraction failed"
    print(f"✓ Center extraction verified")
    
    print("\nAll tests passed!")
    return True


if __name__ == "__main__":
    test_extract_2mhz()