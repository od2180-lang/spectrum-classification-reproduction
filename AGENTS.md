# Agent Instructions

1. Code for this project will go in the repository at https://github.com/od2180-lang/spectrum-classification-reproduction

2. Large data files should go in remote object storage. Credentials for object storage are in the `.env` file.

3. If not logged into GitHub (`gh auth status`), instruct the user to run `gh auth login` and follow the browser-based authentication flow.

---

## Implementation Steps

### Step 1: `run_tds.py` — Detect Transmissions
- Load each full-band .npy file from `spectrum_bands/`
- Compute noise level from data
- Run energy detection + peak finding to identify individual transmissions
- Cut out each detected transmission → individual .npy files
- Save to `detected_transmissions/` with companion .csv metadata (SNR, frequency)
- Output: 2490 raw transmissions

### Step 2: `apply_hopping.py` — Normalize to 2 MHz
- For each detected transmission:
  - Trim to first 50 time segments
  - If width ≥ 200: crop to center 200 bins
  - If width < 200: pad with zeros to 200 bins
- Output: `hopping_results/` (cropped to 200 bins)

### Step 3: `extract_features_psd.py` — Extract Features
- For each PSD segment (50 × 200), extract 33 tsfresh features per row
- Clean NaN/inf values
- Output: `features_33.csv` (2490 rows × 1650 columns: 50 segments × 33 features)
