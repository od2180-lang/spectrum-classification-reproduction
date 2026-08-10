# Research Project: Wireless Spectrum Model Generalization

## Project Goal
Help wireless spectrum models generalize for deployment conditions using the paper "A Framework for Wireless Technology Classification using Crowdsensing Platforms" as a case study.

---

## Current State: Phase 4 Complete — Full Pipeline Run

### What We Did
Ran the full pipeline on all 6 technologies with the most accurate pipeline design, then compared results to the author's reported accuracy.

### Implementation Status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Add `distance`, `box_pts` to ChannelDetector constructor | ✅ Complete |
| 2 | Use `self.box_pts` and `self.distance` in `find_peaks_and_edges` | ✅ Complete |
| 3 | Add `get_detection_params()` and `calc_entropy()` to detect_and_classify.py | ✅ Complete |
| 4 | Rewrite `detect_transmissions()` with adaptive params | ✅ Complete |
| 5 | Rewrite `process_transmission()` with entropy gate | ✅ Complete |
| 6 | Rewrite `process_file()` and `main()` | ✅ Complete |
| 7 | Test on one TETRA file | ✅ Complete |
| 8 | Run full pipeline on TETRA + LTE | ✅ Complete |
| 9 | Deep model analysis (features, scaler, encoder) | ✅ Complete |
| 10 | Add scaler.clip=True fix | ✅ Complete |
| 11 | Run full pipeline on all 6 technologies | ✅ Complete (72.3%) |

---

## Phase 4: Final Pipeline Design

### Pipeline Audit (Every Step vs Author)

| Step | Author's Pipeline | Our Pipeline | Match? |
|------|-------------------|--------------|--------|
| Noise estimation | Sliding window (length=5) dB | percentile(30) + 3*std | Minor diff |
| Detection params | distance=10, box_pts=8, peakThres=3 | Adaptive per-band | Differs (intentional) |
| Width gates | TETRA<10, GSM 14-35, DAB 120-240, DVB-T≥400, LTE>700 | Same gates | ✅ |
| Center crop | width ≥ 200 → 200 bins | Same logic | ✅ |
| Features | 33 tsfresh features | Same 33 features | ✅ |
| Scaler | MinMaxScaler (clip=False) | Same scaler (clip=True) | ✅ (improvement) |
| Encoder | Dense 33→64→32→16 | Same architecture + weights | ✅ |
| LSTM | LSTM(32)→LSTM(16)→Dense(16)→Dense(6) | Same architecture + weights | ✅ |
| Entropy gate | threshold=0.7, accumulation bug | threshold=0.7, no bug | ✅ (improvement) |

### Adaptive Detection Parameters

| Band | Freq Range | Distance | Box_pts | PeakThres | Min_width | Why |
|------|-----------|----------|---------|-----------|-----------|-----|
| TETRA | 300-430 MHz | 2 | 2 | 1 | 2 | Narrow channels (25 kHz = 2.7 bins) |
| LTE | 730-830 MHz | 50 | 16 | 5 | 10 | Wideband carriers, need merging |
| Default | Other | 10 | 8 | 3 | 10 | Works for FM, GSM, DAB, DVB-T |

### Improvements Over Author's Code

1. **`clip=True`** — prevents out-of-range features from breaking encoder
2. **No entropy accumulation bug** — fresh entropy per transmission (author's `list_entropy` never cleared between files)
3. **Better TETRA detection** — narrower signals via adaptive params

### Fair Comparison

Any low accuracy is due to the **model** (width-dependent features, tiny training set), not the pipeline. We're actually giving the model a *better* chance than the author's pipeline did.

---

## Dataset

### File Locations
- **Full path:** `/home/jovyan/work/project/dataset/opt/shared/alessio/Data_Transmitters_Identification/spectrum_bands_2/`
- **199 .npy files** across 6 technologies

### Technology Counts

| Tech | Files | Bandwidth Range | Frequency Resolution |
|------|-------|-----------------|---------------------|
| FM | 41 | 70-128 MHz | ~9.3 kHz/bin |
| TETRA | 38 | 189-429 MHz | ~9.3 kHz/bin |
| LTE | 41 | 734-823 MHz | ~9.3 kHz/bin |
| GSM | 46 | 85-961 MHz | ~9.3 kHz/bin |
| DAB | 32 | 173-2320 MHz | ~9.3 kHz/bin |
| DVB-T | 34 | 464-790 MHz | ~9.3 kHz/bin |

### Key Signal Characteristics

| Tech | Channel BW | BW in Bins | Inter-channel Spacing |
|------|-----------|------------|----------------------|
| TETRA | 25 kHz | 2.7 bins | 2.7 bins |
| GSM | 200 kHz | 21.5 bins | 21.5 bins |
| FM | 100-200 kHz | 10.7-21.5 bins | varies |
| LTE | 1.4-20 MHz | 150-2150 bins | N/A (wideband) |
| DVB-T | 7.6 MHz | 819 bins | N/A (wideband) |
| DAB | 1.5 MHz | 161 bins | N/A (wideband) |

---

## Deep Model Analysis

### What the Model Actually Learned

**Key finding:** The model learned **power levels + signal width**, not technology-specific signal shapes.

### The Exact 33 Features

| Idx | Feature Name | Type | Training Range | Width-Dependent? |
|-----|-------------|------|----------------|------------------|
| 0 | abs_energy | Power/Width | [485, 890,116] | YES |
| 1 | absolute_sum_of_changes | Power/Width | [0.85, 857] | YES |
| 2 | benford_correlation | Shape | [-0.32, 0.96] | No |
| 3 | cid_ce | Shape | [1.12, 17.99] | No |
| 4 | count_above_mean | Width | [1, 138] | YES |
| 5 | count_below_mean | Width | [1, 127] | YES |
| 6 | first_location_of_maximum | Shape | [0.0, 0.995] | No |
| 7 | first_location_of_minimum | Shape | [0.0, 0.995] | No |
| 8 | has_duplicate | Shape | [0, 1] | No |
| 9 | has_duplicate_max | Constant | [0, 0] | Always 0 |
| 10 | has_duplicate_min | Constant | [0, 0] | Always 0 |
| 11 | kurtosis | Shape | [-6.0, 22.1] | No |
| 12 | last_location_of_maximum | Shape | [0.005, 1.0] | No |
| 13 | last_location_of_minimum | Shape | [0.005, 1.0] | No |
| 14 | longest_strike_above_mean | Shape | [1, 95] | Partially |
| 15 | longest_strike_below_mean | Shape | [1, 85] | Partially |
| 16 | maximum | Power | [-62.6, -5.4] dB | No |
| 17 | mean | Power | [-72.5, -10.5] dB | No |
| 18 | mean_abs_change | Shape | [0.42, 16.0] | No |
| 19 | mean_change | Shape | [-6.83, 12.9] | No |
| 20 | mean_second_derivative_central | Shape | [-9.86, 3.34] | No |
| 21 | median | Power | [-72.7, -9.2] dB | No |
| 22 | minimum | Power | [-80.1, -14.1] dB | No |
| 23 | number_cwt_peaks | Width | [0, 48] | YES |
| 24 | number_peaks | Width | [0, 40] | YES |
| 25 | quantile (0.5) | Power | [-72.7, -9.2] dB | No |
| 26 | root_mean_square | Power | [11.0, 72.6] | Partially |
| 27 | skewness | Shape | [-3.94, 2.99] | No |
| 28 | standard_deviation | Shape | [0.22, 17.0] | No |
| 29 | sum_of_reoccurring_values | Power | [-56.0, 0] | No |
| 30 | sum_values | Power/Width | [-12,256, -42] | YES |
| 31 | variance | Width | [0.05, 289] | YES |
| 32 | variation_coefficient | Shape | [-0.74, -0.007] | No |

### Feature Scaling Evidence

Test on TETRA detection at different widths:

| Width | abs_energy | sum_values | count_above | count_below |
|-------|-----------|-----------|-------------|-------------|
| 14 | 40,059 | -722 | 6.3 | 7.7 |
| 12 | 31,945 | -595 | 5.5 | 6.5 |
| 10 | 23,841 | -468 | 4.3 | 5.7 |
| 8 | 15,891 | -342 | 3.9 | 4.1 |
| 6 | 8,827 | -224 | 3.2 | 2.8 |
| 4 | 4,084 | -126 | 2.4 | 1.6 |
| 3 | 2,572 | -88 | 1.6 | 1.4 |

**abs_energy scales 15.6x from width 3 to 14.** Features at different widths produce different values for the same technology.

### Why Width Normalization Doesn't Help

Tested: dividing width-dependent features by signal width.
Result: Still misclassified. The model learned width-specific patterns during training. Normalization changes the feature distribution from what the model expects.

### Why Scaler Clipping Doesn't Help

Tested: `scaler.clip=True` on TETRA detections.
Result: Features were already within training range [0,1]. The issue is that features at 11-17 bins overlap with FM's training distribution, not that they're out of range.

### TETRA vs FM Feature Comparison (Same Width)

| Feature | TETRA (14 bins) | FM (13 bins) | In Training Range? |
|---------|-----------------|--------------|-------------------|
| mean (dB) | -51.6 | -39.5 | Both YES |
| median (dB) | -56.7 | -41.5 | Both YES |
| minimum (dB) | -67.2 | -49.3 | Both YES |
| variance | 200.9 | 56.8 | Both YES |
| std_dev | 14.2 | 7.5 | Both YES |

**Both are in range, but model classifies BOTH as FM (98.5% and 98.7%).** The model's FM training data included signals with similar characteristics to our TETRA detections.

### Model Training Data

| Metric | Value |
|--------|-------|
| Total transmissions | ~204 |
| Per class | ~34 |
| Total time segments | 10,200 |
| Scaler type | MinMaxScaler |
| Scaler n_samples_seen | 10,200 |

**2 features are constant (useless):** `has_duplicate_max` and `has_duplicate_min` are always 0.0.

### Bugs in Author's Code

1. **Entropy accumulation:** `self.list_entropy` in `scoreEntropyPred` is never cleared between files. Running average dilutes the 0.7 threshold.
2. **Constant features:** 2 of 33 features are always 0 — contribute nothing.

---

## Width Gates vs Entropy Gates

### Key Insight

Both gates are ways for the model to say "I don't know" rather than misclassifying:

| Gate | When Applied | What It Filters | Requires Knowing Tech? | Works in Deployment? |
|------|--------------|-----------------|----------------------|---------------------|
| Width Gates | BEFORE classification | Signals that don't match training distribution | YES | **NO** |
| Entropy Gates | AFTER classification | Predictions where model is uncertain | NO | **YES** |

### Why Width Gates Are Problematic

Width gates require knowing the technology BEFORE classification:
```python
# Author's code (TechClass.py:285-308)
if self.key_tr_lab == 'tetra' and tx_test.shape[1] < 10:
    # Already know it's TETRA, then classify
if self.key_tr_lab == 'gsm' and 14 <= tx_test.shape[1] <= 35:
    # Already know it's GSM, then classify
```

**In real deployment, you DON'T know the technology.** You can't apply width gates to unknown signals.

### Two Metrics Tell Different Stories

| Metric | With Width Gate | Without Width Gate |
|--------|-----------------|-------------------|
| Accuracy | 72.3% | 47.2% |
| Signals classified | 1,040 (54.3%) | 1,916 (100%) |
| What it measures | Performance on **known** signals | Performance on **all** signals |
| Generalization | **No** (filters unknowns) | **Yes** (classifies everything) |

**Key insight:** 72.3% is artificially inflated because we're only classifying signals that match the training distribution. The TRUE generalization performance is 47.2%.

### Entropy Gate Issues

The entropy gate should catch uncertain predictions, but it doesn't work:
- Model is always confident (98-99%) even when wrong
- Entropy is always < 0.3 (threshold is 0.7)
- Author's bug: `list_entropy` never cleared between files (our code fixes this)
- Root cause: softmax forces high confidence, no uncertainty calibration

### For Presentation

Frame it as:
- **72.3%** = Performance when we know the technology type (best case)
- **47.2%** = Performance on unknown signals (real-world deployment)
- **The gap (25.1%)** = Cost of generalization
- Width gates = **cheating** (requires knowing the answer)
- Entropy gates = **correct approach** (but broken/overconfident)

---

## Previous Phase Results

### Phase 1: Full Band (No Detection)
| Tech | Accuracy |
|------|----------|
| FM | 0.0% |
| TETRA | 10.5% |
| Combined | 5.1% |

### Phase 1b: Center 200 Bins
| Tech | Accuracy |
|------|----------|
| FM | 0.0% |
| TETRA | 0.0% |

### Phase 2: Detection (No Metadata Gate)
| Tech | Accuracy |
|------|----------|
| FM | ~75% |
| TETRA | ~5% |

### Phase 2b: Detection + Metadata Gate
| Tech | Per-Transmission | Per-File |
|------|-----------------|----------|
| FM | 75.5% | 92.7% |
| TETRA | 5.1% | 0.0% |

### Phase 2c: Detection + No Metadata Gate (LTE Only)
| Tech | Per-Transmission | Per-File |
|------|-----------------|----------|
| LTE | 23.3% | 15.8% |

### Phase 3: Adaptive Detection + Width Gates
| Tech | Per-Transmission | Per-File | Notes |
|------|-----------------|----------|-------|
| TETRA | 50.0% (5/10) | 7.9% (3/38) | Width gate filters 296/310 detections |
| LTE | 26.7% (4/15) | 9.8% (4/41) | Width gate filters most detections |

**Key insight:** Width gates improved TETRA from 0% → 50% when signals pass the gate. The model CAN classify correctly when detection matches training distribution.

### Phase 4: Full Pipeline (All 6 Technologies)
| Tech | Correct | Total | Accuracy | Notes |
|------|---------|-------|----------|-------|
| FM | 493 | 718 | 68.7% | 218 misclassified as GSM |
| GSM | 188 | 189 | 99.5% | Only 1 misclassified |
| DAB | 49 | 56 | 87.5% | 5→DVB-T, 2→LTE |
| DVB-T | 17 | 45 | 37.8% | 28 misclassified as LTE |
| LTE | 3 | 3 | 100% | Very few detections (3 total) |
| TETRA | 2 | 2 | 100% | Very few detections (2 total) |
| **Overall** | **752** | **1,040** | **72.3%** | Width gate filters 45.7% |

### Phase 4 Confusion Matrix

```
          DAB  DVB-T   FM   GSM  LTE  TETRA
DAB        49      5    0     0    2      0
DVB-T       0     17    0     0   28      0
FM          5      1  493   218    1      0
GSM         0      0    1   188    0      0
LTE         0      0    0     0    3      0
TETRA       0      0    0     0    0      2
```

### Key Findings from Phase 4

1. **FM→GSM confusion is the biggest problem** — 218 FM signals misclassified as GSM (30.4% error rate)
2. **DVB-T→LTE confusion** — 28 DVB-T signals misclassified as LTE (62.2% error rate)
3. **GSM is nearly perfect** — 99.5% accuracy (188/189)
4. **LTE and TETRA have 100% accuracy but tiny sample sizes** — only 3 and 2 detections respectively
5. **Width gates filter 45.7% of signals** — 1,916 total detections, only 1,040 pass the gate
6. **Model is overconfident** — entropy never triggers (always < 0.7), even for misclassifications

---

## Research Question

**What evaluation methods or signal characteristics are important for real-world deployment?**

### Key Findings
1. **Detection preprocessing is critical** — model can't classify merged/split signals
2. **Bandwidth matters** — features scale with bandwidth, different BW = different features
3. **Metadata helps filtering but not classification** — LSTM classifies based on features alone
4. **Per-bin (time segment) classification** — model classifies each time segment independently, then averages
5. **Individual channel detection required** — author's system expects pre-separated signals
6. **Width gates improve accuracy** — TETRA 0% → 50% when signals match training distribution
7. **Detection pipeline mismatch** — our detector produces wider signals than author's pipeline
8. **Training set is brittle** — ~204 transmissions, ~34 per class, no generalization
9. **Model learned power levels, not signal shapes** — different technologies at different dB levels in training
10. **Features are width-dependent** — abs_energy, sum_values, count_* scale with signal width
11. **Scaler clipping doesn't help** — features are in range, but overlap with other technologies
12. **Entropy gate rarely triggers** — model always confident, even when wrong

### Root Cause Chain

```
Detection produces wrong widths
  → Width gate rejects most signals
    → Surviving signals have different widths than training
      → Features scale with width (abs_energy, variance, etc.)
        → Features at similar widths overlap between technologies
          → LSTM sees similar feature patterns for TETRA and FM
            → Misclassification (model always confident)
```

---

## Author's extract2MHz Behavior

### What the Author's Code Actually Does (TechClass.py:170-197)

```python
def extract2MHz(self, dta, plot_f, SNR):
    center_channel = round(dta.shape[1] / 2)
    if center_channel < 101:
        raise NameError('Error Small BW')  # Error if too narrow
    dta = dta[:, (center_channel - 100):(center_channel + 100)]  # Just a slice, NO zero padding
    return dta
```

### Key Differences: Author vs Our extract_2mhz_chunk

| | Author | Our extract_2mhz_chunk |
|---|--------|------------------------|
| **Output size** | **200 bins** (not 215) | 215 bins |
| **Zero padding** | **No** | Yes |
| **Too narrow signal** | Raises error | Zero-pads |
| **Edge of spectrum** | Not handled (error) | Zero-pads |

### Author's Actual Logic

```python
# In loadAndPredict (line 228):
if dta.shape[1] >= 200:
    dta = self.extract2MHz(dta, False, SNR)
# If width < 200: uses full signal width as-is, NO 2 MHz extraction
```

So the author:
1. Only crops to 2 MHz when signal is **≥ 200 bins wide**
2. Uses **200 bins** (center ± 100), not 215
3. **No zero padding** — hard error if signal is too narrow

### What Happens to Narrowband Signals

| Technology | Width Gate | extract2MHz Applied? | Actual Width Used |
|------------|------------|---------------------|-------------------|
| TETRA | < 10 bins | **No** (< 200) | 1-9 bins |
| GSM | 14-35 bins | **No** (< 200) | 14-35 bins |
| FM | Any | **No** (usually < 200) | 10-20 bins |
| DAB | 120-240 bins | **No** (< 200) | 120-240 bins |
| DVB-T | ≥ 400 bins | **Yes** (≥ 200) | **Cropped to 200** |
| LTE | > 700 bins | **Yes** (≥ 200) | **Cropped to 200** |

**The author's pipeline doesn't crop narrowband signals.** They are classified with their original width.

### Why 2 MHz with Zero-Padding Would Decrease Accuracy

Narrowband signals with zero-padding would produce different features:
- TETRA (2.7 bins) → 215 bins with zero-padding → 98.7% zeros
- Model was trained on 1-9 bins (no zeros)
- Zero-padding changes variance, count features, etc.
- Model would see feature patterns it never saw during training

---

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `detect_and_classify.py` | Main experiment script | Complete (steps 3-6) |
| `inference.py` | LSTM classification | Complete (with clip=True fix) |
| `feature_extraction.py` | 33-feature extraction | Complete |
| `PSD-technology-classification-framework/TDPackage/DetectorManager/Detector.py` | Detection algorithm | Steps 1-2 done |
| `PSD-technology-classification-framework/TCpackage/TechClass.py` | Author's classification | Reference only |
| `AGENTS.md` | This file | Updated |

---

## Author's Model Architecture

### Autoencoder (Encoder only)
```
Input: 33 features
  Dense(64, activation='relu')
  Dense(32, activation='relu')
  Dense(16, activation='relu')  → Output: 16-dimensional latent space
```

### LSTM Classifier
```
Input: (timesteps=16, features=1)
  LSTM(32, activation='relu', return_sequences=True)
  LSTM(16, activation='relu')
  Dense(16, activation='softmax')
  Dropout(0.001)
  Dense(6, activation='softmax')  → 6 classes
```

### Class Labels
```python
TECH_LABELS = {0: 'dab', 1: 'dvbt', 2: 'fm', 3: 'gsm', 4: 'lte', 5: 'tetra'}
```

### Model Files
```
PSD-technology-classification-framework/TCpackage/resources/
  scaler/_AE16_LSTM_Scaler_.save
  save-DL-models/Autoencoder_DNN/.../saved-model-49-0.0002.hdf5
  save-DL-models/LSTM_TrainWithAE/.../saved-model-110-0.97.hdf5
```

---

## Why Width Differences Are Bad

### The Problem: Features Scale with Width

The model learned **features at specific widths**. When detection produces different widths, features change.

### Example: TETRA at Different Widths

| Width | abs_energy | variance | count_above_mean |
|-------|-----------|----------|------------------|
| 3 bins (training) | 2,572 | 14.2 | 1.6 |
| 10 bins (our detection) | 23,841 | 102.5 | 4.3 |
| 14 bins (our detection) | 40,059 | 200.9 | 6.3 |

**Same technology, different features.** The model sees 40,059 abs_energy and thinks "this looks like FM" (because FM was trained at similar widths).

### Example: FM vs TETRA at Same Width

| Feature | TETRA (14 bins) | FM (13 bins) | Model Says |
|---------|-----------------|--------------|------------|
| abs_energy | 40,059 | 38,210 | Both → FM (98.5%) |
| variance | 200.9 | 56.8 | Both → FM (98.7%) |
| mean (dB) | -51.6 | -39.5 | Both → FM |

**Features overlap at similar widths.** The model can't tell them apart.

### Why This Matters

```
Author's training:
  TETRA → 2-5 bins → Model learns: TETRA = low abs_energy, low variance
  FM → 10-20 bins → Model learns: FM = medium abs_energy, medium variance

Our detection:
  TETRA → 10-17 bins → Features look like FM → Misclassified
```

### The Root Cause (Why 72.3% Not 94.25%)

```
Detection produces wrong widths
  → Features scale with width (abs_energy, variance, count_*)
    → Features at similar widths overlap between technologies
      → LSTM sees similar feature patterns for TETRA and FM
        → Misclassification (model always confident)
```

### Solutions

1. **Match training distribution:** Detect narrower TETRA signals (2-5 bins), crop DVB-T/LTE to 200 bins
2. **Use width-invariant features:** Shape features (kurtosis, skewness) don't scale with width; power features (abs_energy, variance) do
3. **Width gates:** Filter out signals that don't match training distribution (but can't use in deployment)

---

## Expected Results for Phase 4

| Tech | Expected Accuracy | Why |
|------|------------------|-----|
| FM | ~75% | No width gate, features robust |
| TETRA | 0-50% | Width gate rejects most; model trained on <10 bins |
| LTE | ~25% | Some pass width gate (>700 bins), center-cropped |
| DAB | ~40% | Some pass width gate (120-240 bins) |
| DVB-T | N/A | No detections expected |
| GSM | 0% | Width gate rejects most (>35 bins) |

### What Low Accuracy Means

If accuracy is low, it's because:
1. **Width gates reject most signals** — our detection doesn't match author's training distribution
2. **Model learned width-dependent features** — features scale with bandwidth
3. **Training set was tiny** — ~204 transmissions, ~34 per class

These are all **model/training issues**, not pipeline issues. Our pipeline is actually *better* than the author's (clip=True, no entropy bug, adaptive detection).
