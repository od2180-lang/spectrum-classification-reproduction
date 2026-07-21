# Agent Instructions

1. Code for this project will go in the repository at https://github.com/od2180-lang/spectrum-classification-reproduction

2. Large data files should go in remote object storage. Credentials for object storage are in the `.env` file.

3. If not logged into GitHub (`gh auth status`), instruct the user to run `gh auth login` and follow the browser-based authentication flow.

---

## Project Overview

**Paper:** "A Framework for Wireless Technology Classification using Crowdsensing Platforms" (Scalingi et al., IEEE INFOCOM 2023)

**Goal:** Replicate Figure 11 - 2-Layer LSTM confusion matrix using pre-trained models from the authors.

**Data:** 232 .npy files, 40+ sensors across Europe, 6 technologies (DAB, DVB-T, FM, GSM, LTE, TETRA)

**Pipeline:** Load pre-trained weights → Extract features → Classify → Confusion Matrix

---

## Data Structure

- **Location:** `spectrum_bands/<SensorName>/<Date>/SpectrumBands_<start>_<end>_<tech>_<country>_<start>_<end>.npy`
- **Format:** 2D numpy arrays (time_segments × frequency_bins), float64, values in dB
- **Example shape:** (406, 3764) — 406 time segments, 3764 frequency bins

### Known Technologies (from filename)
- DAB (Digital Audio Broadcasting)
- DVB-T (Digital Video Broadcasting - Terrestrial)
- FM (Frequency Modulation radio)
- GSM (2G mobile)
- LTE (4G mobile)
- TETRA (Professional mobile radio)

---

## Feature Extraction (COMPLETED)

### Script: `extract_features_all.py`
Extracts 33 tsfresh statistical features from all .npy files.

**What it does per file:**
1. Loads the .npy file
2. Trims to first **50 time segments** (matches authors' code: `TechClass.py:240` DEF_NUM_TIMESEGMENTS = 50)
3. Crops to **middle 200 frequency bins** (~2 MHz, centered)
4. Extracts **33 tsfresh features per row** (one feature vector per time segment)
5. Replaces inf/NaN with 0

**Output:** `features_33.csv`
- 11,400 rows (228 files × 50 rows, 4 files skipped for being too narrow)
- 40 columns: 7 metadata + 33 features
- 0 NaN, 0 Inf values

### The 33 Features
```
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
```

---

## Pre-trained Models (from authors)

Located in `PSD-technology-classification-framework/TCpackage/resources/`:

| Model | Path | Architecture |
|-------|------|-------------|
| AutoEncoder | `save-DL-models/Autoencoder_DNN/.../saved-model-49-0.0002.hdf5` | 33→64→32→16 (encoder) |
| LSTM Classifier | `save-DL-models/LSTM_TrainWithAE/.../saved-model-110-0.97.hdf5` | LSTM(32)→LSTM(16)→Dense(6) |
| Scaler | `scaler/_AE16_LSTM_Scaler_.save` | MinMaxScaler (33 features) |

**Classification labels:** `{0: 'dab', 1: 'dvbt', 2: 'fm', 3: 'gsm', 4: 'lte', 5: 'tetra', 6: 'unkn'}`

---

## Key Code References

| What | Where |
|------|-------|
| Original feature extraction | `PSD-technology-classification-framework/TCpackage/TechClass.py:86` (`extract_statitics()`) |
| Original 2MHz crop | `PSD-technology-classification-framework/TCpackage/TechClass.py:170` (`extract2MHz()`) |
| Original NaN cleanup | `PSD-technology-classification-framework/TCpackage/TechClass.py:140` (`refine_df()`) |
| Original DEF_NUM_TIMESEGMENTS | `PSD-technology-classification-framework/TCpackage/TechClass.py:240` (= 50) |
| Our feature extraction script | `extract_features_all.py` |
| Our output | `features_33.csv` |

---

## Environment

- Python packages: `tsfresh==0.21.2`, `pandas==2.2.2`, `numpy==1.26.4`, `tensorflow` (for classification)
- GitHub auth: `gh auth status` — logged in as `od2180-lang`
