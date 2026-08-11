#!/usr/bin/env python3
"""
Step 2: Data Discovery
Find all FM and TETRA files, store paths, and parse ground truth labels.
"""

import os
import subprocess

def find_files(technology):
    """Find all .npy files for a given technology."""
    cmd = f'find dataset/ -name "*.npy" | grep "_{technology}_" | sort'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    files = result.stdout.strip().split('\n')
    return [f for f in files if f]  # Filter empty strings

def parse_ground_truth(filepath):
    """Parse technology label from filename."""
    filename = os.path.basename(filepath)
    # Extract technology from filename (e.g., "SpectrumBands_85_105_fm_Esp_85_105.npy" → "fm")
    parts = filename.split('_')
    for part in parts:
        if part.lower() in ['fm', 'tetra', 'dab', 'dvbt', 'gsm', 'lte']:
            return part.lower()
    return None

def main():
    print("=" * 60)
    print("Step 2: Data Discovery")
    print("=" * 60)

    # Find all FM files
    print("\n2.1 Finding FM files...")
    fm_files = find_files('fm')
    print(f"Found {len(fm_files)} FM files")

    # Find all TETRA files
    print("\n2.2 Finding TETRA files...")
    tetra_files = find_files('tetra')
    print(f"Found {len(tetra_files)} TETRA files")

    # Store file paths and parse ground truth
    print("\n2.3 Storing file paths and parsing ground truth...")
    all_files = []
    for f in fm_files:
        gt = parse_ground_truth(f)
        all_files.append({'path': f, 'ground_truth': gt, 'technology': 'fm'})
    for f in tetra_files:
        gt = parse_ground_truth(f)
        all_files.append({'path': f, 'ground_truth': gt, 'technology': 'tetra'})

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total files: {len(all_files)}")
    print(f"FM files: {len(fm_files)}")
    print(f"TETRA files: {len(tetra_files)}")

    # Verify ground truth parsing
    print("\n2.4 Verifying ground truth parsing...")
    fm_with_gt = [f for f in all_files if f['technology'] == 'fm' and f['ground_truth'] == 'fm']
    tetra_with_gt = [f for f in all_files if f['technology'] == 'tetra' and f['ground_truth'] == 'tetra']
    print(f"FM files with correct ground truth: {len(fm_with_gt)}/{len(fm_files)}")
    print(f"TETRA files with correct ground truth: {len(tetra_with_gt)}/{len(tetra_files)}")

    # Show sample files
    print("\nSample FM files:")
    for f in fm_files[:3]:
        print(f"  {os.path.basename(f)}")

    print("\nSample TETRA files:")
    for f in tetra_files[:3]:
        print(f"  {os.path.basename(f)}")

    return fm_files, tetra_files, all_files

if __name__ == "__main__":
    fm_files, tetra_files, all_files = main()
