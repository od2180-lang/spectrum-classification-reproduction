# Agent Instructions

1. Code for this project will go in the repository at https://github.com/od2180-lang/spectrum-classification-reproduction

2. Large data files should go in remote object storage. Credentials for object storage are in the `.env` file.

3. If not logged into GitHub (`gh auth status`), instruct the user to run `gh auth login` and follow the browser-based authentication flow.

---

## Project Overview

**Paper:** "A Framework for Wireless Technology Classification using Crowdsensing Platforms" (Scalingi et al., IEEE INFOCOM 2023)

**Goal:** Replicate Figure 11 — 2-Layer LSTM confusion matrix — using the paper's full pipeline.

**Research Focus:** Identify why wireless spectrum classification models don't generalize to deployment.

---

## Paper's 3-Stage Pipeline

```
Stage 1: Transmission Detection (TDS)
  Full-band .npy capture (thousands of bins)
  → Detect individual transmissions
  → Output: PSD segments (~215 bins each)

Stage 2: Feature Extraction (hopping strategy)
  PSD segment (variable width B)
  → Trim to first 50 time segments
  → Crop to center 200 bins (if B ≥ 200) or pad with zeros (if B < 200)
  → Extract 33 tsfresh features per row
  → Output: 50 × 33 feature matrix

Stage 3: Classification
  50 × 33 features
  → Scale with MinMaxScaler (33 → 33)
  → Encode with AutoEncoder (33 → 16)
  → Classify with 2-Layer LSTM (50 × 16 → 50 × 6 probabilities)
  → Entropy gate (threshold 0.7)
  → Majority vote → final label
```

---

## Data

- **Location:** `spectrum_bands/<SensorName>/<Date>/SpectrumBands_<start>_<end>_<tech>_<country>_<start>_<end>.npy`
- **Format:** 2D numpy arrays (time_segments × frequency_bins), float64, values in dB
- **Technologies:** DAB, DVB-T, FM, GSM, LTE, TETRA

---

## TDS Parameters (from paper + TDTCApp.py)

| Parameter | Value | Source |
|-----------|-------|--------|
| noiseThres | 5 dB | TDTCApp.py:86 |
| peakThres | 3 dB | TDTCApp.py:86 |
| cv | 1 | TDTCApp.py:153 |
| smoothing | True | TDTCApp.py:153 |
| k (prominence) | 0.2 | TDTCApp.py:153 |
| widthApplicable | True | TDTCApp.py:153 |

---

## Hopping Strategy (from paper Section VII-B-3)

1. Detect transmission → PSD segment of width B bins
2. Trim to first **50 time segments**
3. If B ≥ 200: crop to **center 200 bins** (`extract2MHz`, TechClass.py:170)
4. If B < 200: **pad with zeros** to reach 200 bins
5. Result: every input is **50 × 200**

---

## The 33 Features (from TechClass.py:90-128)

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

Note: Paper says 32 features but code has 33. Pre-trained models expect 33.

---

## Pre-trained Models

| Model | Path | Architecture |
|-------|------|-------------|
| AutoEncoder | `PSD-technology-classification-framework/TCpackage/resources/save-DL-models/Autoencoder_DNN/TrainAllSensorHop_16_feat_mse_relu/saved-model-49-0.0002.hdf5` | 33→64→32→16 (encoder) |
| LSTM Classifier | `PSD-technology-classification-framework/TCpackage/resources/save-DL-models/LSTM_TrainWithAE/TrainAllSensorHop_DNNAE16_LSTM_mse_relu/saved-model-110-0.97.hdf5` | LSTM(32)→LSTM(16)→Dense(6) |
| Scaler | `PSD-technology-classification-framework/TCpackage/resources/scaler/_AE16_LSTM_Scaler_.save` | MinMaxScaler (33 features) |

**Labels:** `{0: 'dab', 1: 'dvbt', 2: 'fm', 3: 'gsm', 4: 'lte', 5: 'tetra', 6: 'unkn'}`

---

## Implementation Steps

### Step 1: `run_tds.py` — Detect Transmissions
- Load each full-band .npy file
- Run `ChannelDetector.tx_detection_funct()` with paper's parameters
- Cut out each detected transmission → individual PSD segments
- Save each as separate .npy file

### Step 2: `apply_hopping.py` — Normalize to 2 MHz
- For each detected transmission:
  - Trim to first 50 time segments
  - If width ≥ 200: crop to center 200 bins
  - If width < 200: pad with zeros to 200 bins
- Save as `psd_segments.csv` (metadata + 200 bin values)

### Step 3: `extract_features_psd.py` — Extract Features
- For each PSD segment (50 × 200), extract 33 tsfresh features per row
- Clean NaN/inf values
- Save as `features_33.csv`

### Step 4: `scale_features.py` — Scale (reuse existing)
- Input: `features_33.csv`
- Output: `features_33_scaled.csv`

### Step 5: `encode_features.py` — Encode (reuse existing)
- Input: `features_33_scaled.csv`
- Output: `features_16.csv`

### Step 6: `classify_lstm.py` — Classify (reuse existing)
- Input: `features_16.csv`
- Output: `lstm_predictions.csv`

### Step 7: Evaluate
- Generate confusion matrix
- Report per-class accuracy
- Compare with paper's Figure 11

---

## Key Code References

| What | Where |
|------|-------|
| TDS (transmission detection) | `PSD-technology-classification-framework/TDPackage/DetectorManager/Detector.py` |
| TDS parameters | `PSD-technology-classification-framework/FlaskApp/TDTCApp.py:82-86` |
| Store transmissions | `PSD-technology-classification-framework/TDPackage/Utils/chDetUtils.py:194-240` |
| Noise level computation | `PSD-technology-classification-framework/TDPackage/Utils/chDetUtils.py:99-120` |
| Feature extraction | `PSD-technology-classification-framework/TCpackage/TechClass.py:86-128` |
| 2MHz crop | `PSD-technology-classification-framework/TCpackage/TechClass.py:170-197` |
| NaN cleanup | `PSD-technology-classification-framework/TCpackage/TechClass.py:140-146` |
| LSTM architecture | `PSD-technology-classification-framework/TCpackage/TechClass.py:63-73` |
| Entropy gate | `PSD-technology-classification-framework/TCpackage/TechClass.py:148-168` |
| Classification labels | `PSD-technology-classification-framework/TCpackage/TechClass.py:30` |

---

## Environment

- Python packages: `tsfresh==0.21.2`, `pandas==2.2.2`, `numpy==1.26.4`, `tensorflow`, `scipy`
- GitHub auth: `gh auth status` — logged in as `od2180-lang`
