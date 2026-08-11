#!/usr/bin/env python3
"""
Analyze the .npy dataset using paper-style segment definitions.

Paper definition:
- 1 segment = 1 PSD measurement (single row of spectrogram, ~215 bins)
- Timing: 870 segments × 215 bins = ~40 seconds → 1 segment ≈ 0.046 seconds
- Hopping strategy: For bandwidth > 2 MHz, split into 2 MHz chunks
  - Narrowband (< 2 MHz): 1 chunk per time segment
  - Wideband (≥ 2 MHz): floor(bandwidth_MHz / 2) chunks per time segment

Reference: "A Framework for Wireless Technology Classification
            using Crowdsensing Platforms"
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
import json

DATASET_DIR = Path('/home/jovyan/work/project/dataset/opt/shared/alessio/Data_Transmitters_Identification/spectrum_bands_2')
SEGMENT_DURATION_SEC = 40.0 / 870.0  # Paper: 870 segments × 215 bins = ~40 seconds


def parse_filename(fname):
    """Parse frequency range from filename, handling common typos."""
    parts = fname.replace('.npy','').split('_')
    try:
        tech_idx = None
        for i, p in enumerate(parts):
            if p.lower() in ['tetra', 'gsm', 'lte', 'fm', 'dab', 'dvbt']:
                tech_idx = i
                break
        if tech_idx is None:
            return None, None
        # Use the second frequency pair (after technology) - more reliable
        freq_start = int(parts[tech_idx + 2])
        freq_end = int(parts[tech_idx + 3])
        if freq_start > freq_end:
            # Try first pair instead
            freq_start = int(parts[1])
            freq_end = int(parts[2])
        return freq_start, freq_end
    except:
        return None, None


def main():
    all_files = sorted(DATASET_DIR.rglob('*.npy'))

    # Technology mapping
    tech_map = {}
    for f in all_files:
        fname = f.name.lower()
        for tech in ['tetra', 'gsm', 'lte', 'fm', 'dab', 'dvbt']:
            if f'_{tech}_' in fname:
                tech_map[f] = tech
                break

    results = []
    tech_stats = defaultdict(lambda: {
        'files': 0, 'paper_segments': 0, 'time_segments': 0,
        'total_chunks': 0, 'bandwidths': []
    })
    grand_total_paper_segs = 0
    grand_total_time_segs = 0
    empty_count = 0

    for tech in ['dab', 'dvbt', 'fm', 'gsm', 'lte', 'tetra']:
        for f in sorted([k for k, v in tech_map.items() if v == tech]):
            arr = np.load(f)
            freq_bins, time_segs = arr.shape
            if time_segs == 0:
                empty_count += 1

            freq_start, freq_end = parse_filename(f.name)
            if freq_start is None or freq_end is None:
                continue
            bandwidth_mhz = freq_end - freq_start
            if bandwidth_mhz <= 0:
                continue

            # Paper's hopping strategy
            if bandwidth_mhz <= 2:
                num_chunks = 1
            else:
                num_chunks = int(bandwidth_mhz // 2)

            paper_segs = time_segs * num_chunks
            duration_sec = paper_segs * SEGMENT_DURATION_SEC
            duration_min = duration_sec / 60.0
            if duration_min >= 60:
                duration_str = f"{duration_min/60:.2f} hrs"
            else:
                duration_str = f"{duration_min:.1f} min"

            sensor = f.parent.parent.name
            results.append({
                'tech': tech, 'bw': bandwidth_mhz, 'freq_bins': freq_bins,
                'time_segs': time_segs, 'chunks': num_chunks,
                'paper_segs': paper_segs, 'duration': duration_str,
                'filename': f.name, 'sensor': sensor
            })

            tech_stats[tech]['files'] += 1
            tech_stats[tech]['paper_segments'] += paper_segs
            tech_stats[tech]['time_segments'] += time_segs
            tech_stats[tech]['total_chunks'] += num_chunks
            tech_stats[tech]['bandwidths'].append(bandwidth_mhz)
            grand_total_paper_segs += paper_segs
            grand_total_time_segs += time_segs

    # Print summary
    print(f"Total files: {len(results)}")
    print(f"Total time segments: {grand_total_time_segs:,}")
    print(f"Total paper-style segments: {grand_total_paper_segs:,}")
    grand_duration_hrs = (grand_total_paper_segs * SEGMENT_DURATION_SEC) / 3600.0
    print(f"Total duration: {grand_duration_hrs:.2f} hours")
    print(f"Empty files: {empty_count}")

    for tech in ['dab', 'dvbt', 'fm', 'gsm', 'lte', 'tetra']:
        s = tech_stats[tech]
        avg_chunks = s['total_chunks'] / s['files'] if s['files'] > 0 else 0
        avg_bw = sum(s['bandwidths']) / len(s['bandwidths']) if s['bandwidths'] else 0
        duration_sec = s['paper_segments'] * SEGMENT_DURATION_SEC
        duration_min = duration_sec / 60.0
        if duration_min >= 60:
            duration_str = f"{duration_min/60:.2f} hrs"
        else:
            duration_str = f"{duration_min:.1f} min"
        print(f"{tech.upper()}: {s['files']} files, {s['time_segments']:,} time segs, "
              f"avg {avg_chunks:.1f} chunks, {s['paper_segments']:,} paper segs, "
              f"{duration_str}, avg BW {avg_bw:.1f} MHz")


if __name__ == '__main__':
    main()
