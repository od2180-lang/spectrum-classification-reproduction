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
  → Output: PSD segments (variable width B)

Stage 2: Feature Extraction (hopping strategy)
  PSD segment (variable width B)
  → Trim to first 50 time segments
  → WIDTH GATE checks ORIGINAL width (before cropping)
  → Crop to center 200 bins (if B ≥ 200)
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

- **Raw transmissions:** `detected_transmissions/<filename>.npy` + `<filename>.csv` (metadata with SNR)
- **Cropped transmissions:** `hopping_results/<filename>.npy` (already cropped to 200 bins — DO NOT use for authors' pipeline)
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

---

## Accuracy Results

### Authors' Pre-trained Models (replicate_figure11.py)
Uses authors' exact `TechClass.py` code (3 minimal patches: dead import, model path, lr deprecation).

| Metric | Value |
|--------|-------|
| **Overall Accuracy (entropy gate)** | **68.42%** (221/323 known predictions) |
| **Unknown filtered by entropy gate** | 2167/2490 (87%) |
| **Paper's reported accuracy** | **94.25%** |

**Gap to paper:** -25.83%. Caused by:
1. Buggy frequency ID — DVB-T/GSM/LTE always become `'unkn'` (`*1e6` on both sides)
2. `list_entropy` accumulation — gate filters 87% as unknown, compounds across files
3. `h + 0.7` on low-SNR path — inflates accumulated entropy further
4. LSTM bias toward DAB — model predicts DAB for most things (217/323 predictions are DAB)

**Note:** This IS the authors' exact code. The gap likely means they ran on a different/smaller test set where the accumulation bug didn't compound as severely.

### Train-from-Scratch (classify_lstm.py)
Trains own LSTM from scratch (no pre-trained models). Uses `hopping_results/` cropped data.

| Metric | Value |
|--------|-------|
| **Overall Accuracy (entropy gate)** | **95.24%** (140/147 confident) |
| **Overall Accuracy (no gate)** | 75.90% (378/498) |
| **Paper's reported accuracy** | **94.25%** |

**Note:** Train-from-scratch beats the paper but only on 147 confident predictions (91% filtered as unknown). Without gate, drops to 75.90%.

### Per-Class Accuracy (Authors' Pre-trained, with Entropy Gate)

| Technology | Correct | Total | Accuracy | Paper |
|------------|---------|-------|----------|-------|
| DAB | 217 | 217 | 100.00% | 98% |
| DVB-T | 0 | 1 | 0.00% | 93% |
| FM | 4 | 6 | 66.67% | 97% |
| GSM | 0 | 0 | - | 98% |
| LTE | 0 | 0 | - | 98% |
| TETRA | 0 | 99 | 0.00% | 97% |

*Note: list_entropy accumulation bug causes 87% of files (2167/2490) to be filtered as unknown, leaving only 323 confident predictions. DVB-T/GSM/LTE always become 'unkn' due to buggy frequency identification. TETRA files pass the gate but LSTM predicts DAB for all of them.*

---

## Authors' Exact Pipeline (TechClass.py)

The full classification flow for each transmission:

```
inference_trx_labels() → for each .npy file:
  1. Load raw .npy → (50+, width)
  2. Trim to 50 time segments → (50, width)
  3. Read SNR from .csv sidecar file
  4. Determine technology from frequency range
  5. WIDTH GATE checks ORIGINAL width (see below)
  6. If passes gate → loadAndPredict()

loadAndPredict():
  7. Load raw .npy AGAIN → (50+, width)
  8. Trim to 50 time segments → (50, width)
  9. If width ≥ 200: crop to CENTER 200 bins (extract2MHz)
  10. Extract 33 features → (50, 33)
  11. Clean NaN/inf (refine_df: fill skewness, kurtosis, mean_second_derivative_central with 0)
  12. Remove id_sensor column
  13. → inference_data()

inference_data():
  14. Scale with authors' MinMaxScaler → (50, 33)
  15. Encode with authors' AutoEncoder encoder → (50, 16)
  16. Reshape each segment to (16, 1) → (50, 16, 1)
  17. Classify each segment with authors' LSTM → (50, 6) probabilities
  18. If SNR > 3: entropy gate + majority vote
  19. If SNR ≤ 3: force label 6 (unknown)
  20. Return final label
```

---

## Width Gates (CRITICAL — checks ORIGINAL width before cropping)

From TechClass.py:285-312:

| Technology | Width Condition | SNR Condition |
|------------|----------------|---------------|
| DAB | 120 ≤ width ≤ 240 | SNR ≥ 1.70 |
| DVB-T | width ≥ 400 | none |
| FM | no gate | none |
| GSM | 14 ≤ width ≤ 35 | none |
| LTE | width > 700 | none |
| TETRA | width < 10 | none |

**CRITICAL: Width gate checks the RAW transmission width, NOT the cropped width.**

Our data widths (raw from `detected_transmissions/`):
- DAB: 9-1400 bins (82/371 pass gate)
- DVB-T: 9-6417 bins (81/310 pass gate)
- FM: 7-1219 bins (1046/1046 pass — no gate)
- GSM: 11-642 bins (168/327 pass gate)
- LTE: 12-3371 bins (99/165 pass gate)
- TETRA: 5-177 bins (100/271 pass gate)

---

## Entropy Calculation Discrepancy (CRITICAL)

**Issue 1: Different math**

Authors' code (TechClass.py:155-160):
```python
def scoreEntropyPred(self, test_Y_i_hat, treshold_alpha):
    for elem in test_Y_i_hat:           # for each of 50 segment predictions
        h = self.calc_entropy(elem)      # entropy of ONE segment's 6-class probs
        self.list_entropy.append(h)
    entropy_avg = np.mean(self.list_entropy)  # average of 50 individual entropies
```

Our code (WRONG):
```python
prob_mean = np.mean(file_probs, axis=0)   # average the 50 predictions FIRST
entropy = -np.sum(prob_mean * np.log2(prob_mean + 1e-10))  # entropy of the average
```

**These are mathematically different.** The authors compute 50 individual entropies and average them. We compute the mean probability vector and take its entropy.

**Issue 2: self.list_entropy accumulates across files**

```python
class TechnologyClassifClass:
    def __init__(self):
        self.list_entropy = []   # initialized ONCE (line 33)

    def scoreEntropyPred(self, test_Y_i_hat, treshold_alpha):
        for elem in test_Y_i_hat:
            h = self.calc_entropy(elem)
            self.list_entropy.append(h)    # APPENDS, never cleared
        entropy_avg = np.mean(self.list_entropy)
```

`self.list_entropy` is NEVER cleared between files. Entropies from ALL previous files accumulate, making the gate increasingly aggressive as more files are processed. This is likely a bug, but it's what the code does.

---

## Key Code References

| What | Where |
|------|-------|
| TDS (transmission detection) | `PSD-technology-classification-framework/TDPackage/DetectorManager/Detector.py` |
| TDS parameters | `PSD-technology-classification-framework/FlaskApp/TDTCApp.py:82-86` |
| Store transmissions | `PSD-technology-classification-framework/TDPackage/Utils/chDetUtils.py:194-240` |
| Noise level computation | `PSD-technology-classification-framework/TDPackage/Utils/chDetUtils.py:99-120` |
| Feature extraction | `PSD-technology-classification-framework/TCpackage/TechClass.py:86-128` |
| 2MHz crop (center) | `PSD-technology-classification-framework/TCpackage/TechClass.py:170-197` |
| NaN cleanup | `PSD-technology-classification-framework/TCpackage/TechClass.py:140-146` |
| LSTM architecture | `PSD-technology-classification-framework/TCpackage/TechClass.py:63-73` |
| Entropy gate | `PSD-technology-classification-framework/TCpackage/TechClass.py:148-168` |
| Width gates | `PSD-technology-classification-framework/TCpackage/TechClass.py:285-312` |
| Classification labels | `PSD-technology-classification-framework/TCpackage/TechClass.py:30` |

---

## Environment

- Python packages: `tsfresh==0.21.2`, `pandas==2.2.2`, `numpy==1.26.4`, `tensorflow`, `scipy`
- GitHub auth: `gh auth status` — logged in as `od2180-lang`
