# Agent Instructions

1. Code for this project will go in the repository at https://github.com/od2180-lang/spectrum-classification-reproduction

2. Large data files should go in remote object storage. Credentials for object storage are in the `.env` file.

3. If not logged into GitHub (`gh auth status`), instruct the user to run `gh auth login` and follow the browser-based authentication flow.

---

## Project Overview

**Paper:** "A Framework for Wireless Technology Classification using Crowdsensing Platforms" (Scalingi et al., IEEE INFOCOM 2023)

**Goal:** Replicate Figure 11 - 2-Layer LSTM confusion matrix using pre-trained models from the authors.

**Data:** 232 .npy files, 40+ sensors across Europe, 6 technologies (DAB, DVB-T, FM, GSM, LTE, TETRA)

**Pipeline:** Load pre-trained weights → Extract features → Scale features → Classify → Confusion Matrix

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

## Feature Scaling (COMPLETED)

### Script: `scale_features.py`
Scales the 33 extracted features using the authors' pre-trained MinMaxScaler.

**What it does:**
1. Loads the pre-trained scaler from `PSD-technology-classification-framework/TCpackage/resources/scaler/_AE16_LSTM_Scaler_.save`
2. Reads `features_33.csv`
3. Applies `scaler.transform()` on all 33 feature columns (never `.fit()`)
4. Saves `features_33_scaled.csv`

**Key details:**
- Scaler type: `MinMaxScaler` (scikit-learn 1.0.2), feature_range=(0, 1)
- Fitted on the original training set's 33 features
- Some scaled values fall outside [0, 1] (range: -1.26 to 1.80) — this is normal because our data extrapolates beyond training min/max. The downstream models (AutoEncoder, LSTM) handle this fine
- Feature column order matches `TechClass.py:86-128` exactly

**Output:** `features_33_scaled.csv`
- 11,400 rows, 40 columns (7 metadata + 33 scaled features)
- 0 NaN, 0 Inf values

### Authors' scaling approach (TechClass.py)
The authors scale each time segment independently:
```python
# TechClass.py:199-223 (inference_data)
X_test = self.scaler.transform(X)        # X is (50, 33)
X_test_encode = encoder.predict(X_test)  # 33 → 16
X_test_encode = np.reshape(X_test_encode, (-1, 16, 1))  # LSTM input
test_Y_i_hat = model.predict(X_test)     # classify
```
- All 50 rows per file are scaled, encoded, and classified independently
- Final label determined by entropy-based aggregation (`scoreEntropyPred`, line 155)

---

## AutoEncoder Encoding (COMPLETED)

### Script: `encode_features.py`
Encodes the 33 scaled features into 16 compressed features using the authors' pre-trained AutoEncoder.

**What it does:**
1. Builds the autoencoder architecture (encoder: 33→64→32→16, decoder: 16→32→64→33)
2. Loads pre-trained weights from `saved-model-49-0.0002.hdf5`
3. Extracts only the encoder half (discards decoder)
4. Reads `features_33_scaled.csv`
5. Applies `encoder.predict()` on all 11,400 rows
6. Saves `features_16.csv`

**Key details:**
- Encoder architecture matches `TechClass.py:46-50` exactly
- Activation function: `relu` (line 78)
- Each row's 33 features → 16 compressed features (independent processing)
- No NaN or Inf values in output

**Output:** `features_16.csv`
- 11,400 rows, 23 columns (7 metadata + 16 encoded features)
- Feature columns named `enc_0` through `enc_15`
- Value range: 0.0 to 4.87 (relu activation output)

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
| Our feature scaling script | `scale_features.py` |
| Our autoencoder encoding script | `encode_features.py` |
| Our output (unscaled) | `features_33.csv` |
| Our output (scaled) | `features_33_scaled.csv` |
| Our output (encoded) | `features_16.csv` |

---

## Environment

- Python packages: `tsfresh==0.21.2`, `pandas==2.2.2`, `numpy==1.26.4`, `tensorflow` (for classification)
- GitHub auth: `gh auth status` — logged in as `od2180-lang`
