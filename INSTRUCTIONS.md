# Task: Fix the Figure 11 reproduction pipeline

Work directly in the `spectrum-classification-reproduction` repository.

The public dataset is already downloaded and extracted under:

```text
data/
```

The pretrained scaler, autoencoder, and LSTM are under:

```text
PSD-technology-classification-framework/TCpackage/resources/
```

## Objective

Fix the existing code so that `replicate_figure11.py` evaluates the pretrained model as closely as possible to the closed-set experiment used for Figure 11 of:

> A Framework for Wireless Technology Classification using Crowdsensing Platforms

The final output must be a six-class confusion matrix with this class order:

```text
DAB, DVB-T, FM, GSM, LTE, TETRA
```

The paper reports:

- 108,000 training PSD segments
- 26,490 testing PSD segments
- Equal class balance
- 18,000 training segments per class
- 4,415 testing segments per class
- 94.25% overall test accuracy
- Hopping preprocessing starting at the first frequency bin
- At most 2 MHz per feature vector
- One prediction per PSD segment

The rounded Figure 11 matrix is approximately:

| True \ Predicted | DAB | DVB-T | FM | GSM | LTE | TETRA |
|---|---:|---:|---:|---:|---:|---:|
| DAB | 0.98 | 0.01 | 0.00 | 0.00 | 0.01 | 0.00 |
| DVB-T | 0.00 | 0.87 | 0.00 | 0.00 | 0.13 | 0.00 |
| FM | 0.00 | 0.00 | 0.95 | 0.05 | 0.00 | 0.00 |
| GSM | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| LTE | 0.01 | 0.08 | 0.00 | 0.00 | 0.90 | 0.00 |
| TETRA | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.99 |

Do not hardcode these values. They are only a diagnostic target. The rounded cells do not reproduce the exact reported 94.25% when averaged.

## Important constraints

Keep the solution small and focused.

- Fix the existing scripts. Do not create a new framework.
- Do not add a test suite, database, pipeline engine, configuration framework, or extensive abstraction layer.
- Use the released pretrained scaler, autoencoder, and LSTM.
- Do not retrain the model unless explicitly requested later.
- Do not introduce a leakage-resistant grouped split.
- Do not group by sensor, site, source file, date, or detected transmission.
- If the original work split individual PSD segments after preprocessing, preserve that behavior.
- Adjacent rows or hopping chunks from the same source may therefore occur in both the conceptual training and testing subsets.
- Do not “correct” this possible leakage for the primary reproduction.
- Do not brute-force random seeds until the result matches 94.25%.
- Do not tune detector thresholds or preprocessing based directly on the published confusion-matrix values.
- Do not hide skipped samples or silently filter difficult predictions.
- Do not claim exact reproduction if the original test membership cannot be recovered.

## References

- Paper: https://ieeexplore.ieee.org/document/10228867
- Original framework: https://github.com/electrosense/PSD-technology-classification-framework
- Public dataset: https://zenodo.org/records/7521246
- Current reproduction: https://github.com/od2180-lang/spectrum-classification-reproduction

---

# Required investigation

Before editing, inspect the repository and dataset:

```bash
git status --short

find data -type f -name '*.npy' | head -20
find data -type f -name '*.npy' | wc -l

find PSD-technology-classification-framework/TCpackage/resources -type f | sort

grep -R "define_noise_level\|tx_detection_funct\|ChannelDetector" \
  PSD-technology-classification-framework/TDPackage

grep -R "train_test_split\|random_state\|shuffle\|18000\|4415\|26490\|108000" \
  PSD-technology-classification-framework .
```

Print the number of dataset files found for each class.

Load several representative files and print:

- Filename
- Shape
- Data type
- Minimum
- Maximum
- Whether frequency is the second dimension
- Parsed frequency range
- Expected number of frequency bins based on approximately 10 kHz/bin

Do not assume the arrays need transposition. Only transpose if inspection clearly proves that a file is oriented as frequency-by-time instead of time-by-frequency.

Ground truth must come from the technology label in the path or filename, not from the frequency range.

Use a label expression equivalent to:

```python
r"_(dab|dvbt|fm|gsm|lte|tetra)(?:_|\.|$)"
```

The start frequency may still be used to choose transmission-detector parameters. It must not determine the true label.

---

# Main pipeline corrections

## 1. Stop using the deployment pipeline for Figure 11

The current `replicate_figure11.py` copies the deployment-oriented `TechClass.py` behavior. That is not the Figure 11 experiment.

For the Figure 11 path, do not use:

- Frequency-based technology gates
- Class-specific width gates
- SNR rejection
- Entropy thresholding
- An unknown class
- Center cropping
- Averaging predictions over 50 rows
- Majority voting over a detected transmission
- One prediction per `.npy` transmission file
- Filtering unknown predictions before scoring

It is acceptable to replace most of `replicate_figure11.py` with a simpler script that imports the existing detector, feature extractor, and model loader.

Do not keep duplicated copies of the model and feature-extraction implementations when the existing modules can be imported.

The intended imports should be approximately:

```python
from detect_and_classify import (
    parse_frequency,
    noise_estimation,
    detect_transmissions,
)
from feature_extraction import extract_33_features
from inference import load_models, predict_segment_probabilities
```

If the reproduction's `noise_estimation()` is only an approximation and the original framework contains the real noise-level method, use the original method instead. Reuse the existing implementation from `TDPackage`; do not write another detector.

## 2. Preserve the checkpoint’s actual LSTM input

Do not create temporal LSTM sequences from consecutive PSD rows.

The released checkpoint expects this exact transformation:

```text
one PSD row or hopping chunk
    -> 33 statistical features
    -> scaler
    -> 16 autoencoder latent values
    -> reshape to (16, 1)
    -> two-layer LSTM
    -> six probabilities
```

For a batch:

```python
features.shape == (N, 33)
encoded.shape == (N, 16)
lstm_input.shape == (N, 16, 1)
probabilities.shape == (N, 6)
```

The 16 latent feature coordinates are being treated as the LSTM sequence. Do not change this architecture.

Do not combine 50 PSD rows into one LSTM sample.

## 3. Fix `inference.py`

Remove this line:

```python
scaler.clip = True
```

The original inference code loads the serialized scaler and directly calls `transform`. Setting `clip=True` changes its behavior.

Keep the exact released architecture and weights.

Add a small function that returns one probability vector per input PSD segment:

```python
def predict_segment_probabilities(
    features,
    scaler,
    encoder,
    model,
    batch_size=4096,
):
    scaled = scaler.transform(features)
    encoded = encoder.predict(
        scaled,
        batch_size=batch_size,
        verbose=0,
    )
    lstm_input = encoded.reshape(-1, 16, 1)
    probabilities = model.predict(
        lstm_input,
        batch_size=batch_size,
        verbose=0,
    )
    return probabilities
```

Add direct shape checks:

```python
assert features.ndim == 2
assert features.shape[1] == 33
assert encoded.shape[1] == 16
assert probabilities.shape[1] == 6
```

Do not average `probabilities` in this function.

Do not apply temperature scaling.

Do not apply entropy filtering.

The prediction for each segment is:

```python
predicted_class = probabilities.argmax(axis=1)
```

The old `classify()` function may remain for other scripts, but Figure 11 must not use its averaging behavior.

## 4. Preserve the exact feature order

Do not redesign `feature_extraction.py`.

The existing function should produce 33 features in the same order as the original `TechClass.extract_statitics()` method.

Verify:

```python
features.shape[1] == 33
len(columns) == 33
```

Print the complete ordered feature-name list once during startup.

Check the loaded scaler:

```python
if hasattr(scaler, "n_features_in_"):
    assert scaler.n_features_in_ == 33
```

Do not add sensor ID as a model feature. The original code appended it temporarily and removed it before inference.

Do not silently replace every non-finite feature with zero. Preserve the original refinement behavior:

- Replace positive infinity with NaN.
- Fill NaN in `skewness`.
- Fill NaN in `mean_second_derivative_central`.
- Fill NaN in `kurtosis`.

After that, check for remaining non-finite values. If any remain, print:

- Source filename
- Detected transmission bounds
- Hopping bounds
- Row index
- Feature names containing invalid values

Do not silently continue with fabricated values.

## 5. Detect transmissions, but do not use metadata classification gates

The public files contain labeled spectrum bands. Use the existing transmission detector to obtain individual transmission boundaries within each band.

For each source file:

1. Load the PSD matrix.
2. Confirm its orientation.
3. Estimate the noise floor using the original detector implementation.
4. Run the existing `ChannelDetector`.
5. For each detected channel, slice:

```python
transmission = data[:, start_bin:end_bin + 1]
```

The detector output may use frequency-dependent parameters. That is acceptable.

After detection, do not require a detected channel to satisfy expected class bandwidth rules such as:

```text
DAB: 120–240 bins
DVB-T: at least 400 bins
GSM: 14–35 bins
LTE: more than 700 bins
TETRA: less than 10 bins
```

Those rules must not decide whether a known-class test sample is evaluated.

Do not reject based on SNR.

Do not map a band to DVB-T, LTE, or another class based on its frequency. The filename/path label is the ground truth.

## 6. Implement the paper’s hopping preprocessing

Delete or bypass the center-crop behavior:

```python
center = round(width / 2)
tx[:, center - 100:center + 100]
```

Use sequential hopping from the first frequency bin.

Add one small helper, either in `replicate_figure11.py` or `detect_and_classify.py`:

```python
def iter_hopping_chunks(transmission, chunk_bins=200):
    width = transmission.shape[1]

    if width <= chunk_bins:
        yield 0, width, transmission
        return

    for start in range(0, width, chunk_bins):
        end = min(start + chunk_bins, width)
        chunk = transmission[:, start:end]

        if chunk.shape[1] > 1:
            yield start, end, chunk
```

Important behavior:

- A narrow transmission produces one chunk using its full bandwidth.
- A wide transmission produces sequential chunks.
- The first chunk begins at frequency bin zero of the detected transmission.
- No chunk is wider than 200 bins.
- There is no center crop.
- There are no overlapping hopping chunks.
- Retain the final partial chunk if it contains enough bins to calculate the features.
- Log how many full and partial chunks are produced.

The paper describes the sensor bandwidth as approximately 2 MHz and roughly 200–215 bins. Use 200 because that is what the released deployment code and current reproduction use, unless direct inspection of the original training artifacts establishes a different exact value.

Do not change the chunk size merely because another value produces a better confusion matrix.

## 7. Make one prediction per PSD row and hopping chunk

For each detected transmission and each hopping chunk:

```python
chunk.shape == (number_of_time_rows, number_of_frequency_bins)
```

Run:

```python
features, columns = extract_33_features(chunk.astype(np.float32))
```

This produces one feature vector for each PSD row.

Every row of `features` is one classification sample.

For example:

```text
50 time rows × 4 hopping chunks = 200 classification samples
```

Do not reduce those 200 samples to one transmission prediction.

Do not truncate every transmission to the first 50 rows for the Figure 11 evaluation. Use all valid rows available in the labeled dataset.

If memory becomes an issue, process one file or transmission at a time and append only the 33-feature arrays. Do not hold the complete raw dataset in memory.

## 8. Recreate the paper-like sample-level split

After all valid feature vectors have been generated, keep them separated by true class.

Print the raw candidate count for every class before balancing.

The paper’s balanced total corresponds to:

```text
22,415 samples per class
18,000 conceptual training samples per class
4,415 testing samples per class
```

Search the original repository and model directories for evidence of:

- A random seed
- A saved permutation
- Test indices
- Training arrays
- Validation arrays
- A sample manifest
- A file ordering convention

Use that information if it exists.

If no split information exists, use this minimal deterministic fallback:

```python
SEED = 42
SAMPLES_PER_CLASS = 22415
TRAIN_PER_CLASS = 18000
TEST_PER_CLASS = 4415
```

For each class:

1. Start with candidates in deterministic order:
   - Sorted source pathname
   - Detector channel order
   - Time-row order
   - Hopping-chunk order
2. Shuffle individual samples using one deterministic NumPy RNG.
3. Keep exactly 22,415 samples.
4. Treat the first 18,000 as the conceptual training portion.
5. Evaluate the remaining 4,415 as the test portion.

Do not group samples before shuffling.

This deliberately permits rows and hopping chunks from the same source transmission to be divided across the conceptual train and test portions. That matches the requested paper-like behavior rather than introducing a grouped leakage fix.

The pretrained checkpoint is already trained, so the conceptual training subset is not used to fit anything. It exists only to recreate the reported test-set size and sample-level selection procedure.

If a class has fewer than 22,415 valid candidates, fail clearly and print the counts. Do not silently lower every class to the smallest class. Investigate preprocessing or detection first.

Do not try many random seeds to select whichever produces 94.25%.

## 9. Build the closed-set confusion matrix

Use this exact class order:

```python
LABELS = ["dab", "dvbt", "fm", "gsm", "lte", "tetra"]
```

For every one of the 26,490 test segments:

```python
predicted_index = np.argmax(probabilities, axis=1)
```

Build a `6 x 6` integer matrix where:

```text
rows = true class
columns = predicted class
```

There must be no unknown class.

There must be no entropy gate.

There must be no rejected prediction.

There must be no filtering before the matrix is calculated.

Normalize each row only for display:

```python
normalized = counts / counts.sum(axis=1, keepdims=True)
```

Calculate overall accuracy from the integer matrix:

```python
accuracy = np.trace(counts) / counts.sum()
```

Do not calculate accuracy from the rounded normalized matrix.

Save:

```text
figures/figure11_reproduced.png
figures/figure11_counts.csv
figures/figure11_normalized.csv
```

The plot must include:

- True labels on the y-axis
- Predicted labels on the x-axis
- Class order matching the paper
- Normalized values printed to two decimal places
- Overall integer-count accuracy in the title
- No “unknown” row or column

---

# Minimal command-line behavior

Add only the command-line options needed to run and inspect the script:

```bash
python replicate_figure11.py --data-dir data
```

A small smoke-run option is useful:

```bash
python replicate_figure11.py \
  --data-dir data \
  --max-files-per-class 1
```

The smoke run does not need to produce the final 26,490-sample matrix. Its purpose is to verify loading, detection, hopping, feature extraction, and model inference.

Do not add a complex configuration system.

---

# Required validation

## Static checks

Run:

```bash
python -m py_compile \
  feature_extraction.py \
  inference.py \
  detect_and_classify.py \
  replicate_figure11.py
```

Search for prohibited Figure 11 behavior:

```bash
grep -n "scaler.clip" inference.py
grep -n "entropy\|threshold_alpha\|unkn" replicate_figure11.py
grep -n "center_channel\|center.*100" replicate_figure11.py
grep -n "mean(axis=0)\|majority" replicate_figure11.py
```

Expected result:

- No `scaler.clip = True`
- No entropy-based decision in the Figure 11 path
- No unknown class in the Figure 11 path
- No center crop in the Figure 11 path
- No averaging that converts several PSD rows into one prediction
- No majority vote in the Figure 11 path

Entropy utilities may remain elsewhere for the separate open-set experiment, but they must not be called by `replicate_figure11.py`.

## Smoke-run checks

Run:

```bash
python replicate_figure11.py \
  --data-dir data \
  --max-files-per-class 1
```

The script must print:

- All six labels discovered
- Loaded model paths
- Ordered list of 33 features
- Input file shapes
- Detected transmission count
- Detected widths
- Hopping bounds
- Number of feature vectors generated
- Feature shape `(N, 33)`
- Encoded shape `(N, 16)`
- LSTM shape `(N, 16, 1)`
- Probability shape `(N, 6)`
- Probability-row sums close to one

Add assertions equivalent to:

```python
assert features.shape[1] == 33
assert encoded.shape[1] == 16
assert lstm_input.shape[1:] == (16, 1)
assert probabilities.shape[1] == 6
assert np.allclose(
    probabilities.sum(axis=1),
    1.0,
    atol=1e-4,
)
```

Manually verify at least one wide detected transmission:

```text
width > 200
first chunk starts at 0
second chunk starts at 200
no center crop
```

Manually verify at least one narrow transmission:

```text
width < 200
one chunk
original width retained
```

## Full-run checks

Run:

```bash
python replicate_figure11.py --data-dir data
```

The final run is valid only if it reports:

```text
Test samples: 26490
DAB:   4415
DVB-T: 4415
FM:    4415
GSM:   4415
LTE:   4415
TETRA: 4415
```

Also verify:

```python
assert counts.shape == (6, 6)
assert counts.sum() == 26490
assert np.all(counts.sum(axis=1) == 4415)
assert np.allclose(normalized.sum(axis=1), 1.0)
```

The number of predictions must equal the number of true labels:

```python
assert len(y_true) == len(y_pred) == 26490
```

There must be no code equivalent to:

```python
known_mask = pred != "unkn"
```

There must be no denominator based only on accepted or confident predictions.

## Compare with the paper

Print the reproduced normalized matrix and the published rounded target matrix.

Also print:

```text
Overall accuracy
Per-class accuracy
Absolute difference from 94.25 percentage points
Mean absolute cell difference from the published rounded matrix
Largest cell difference
```

Use the comparison only as a diagnostic.

The expected qualitative error pattern is:

- DAB is mostly correct.
- DVB-T is mainly confused with LTE.
- FM is mainly confused with GSM.
- GSM is nearly perfect.
- LTE is mainly confused with DVB-T and occasionally DAB.
- TETRA is nearly perfect and is occasionally confused with FM.

If the result differs substantially, investigate in this order:

1. Wrong feature ordering
2. `scaler.clip=True` still active
3. Wrong data orientation
4. Center crop still being used
5. Predictions averaged over rows
6. Consecutive rows incorrectly assembled into temporal sequences
7. Frequency-derived ground truth
8. Classification width gates
9. Entropy or SNR rejection
10. First-50-row truncation
11. Incorrect hopping boundaries
12. Approximate rather than original noise estimation
13. Custom filtering of detector output
14. Incorrect sample balancing
15. Unrecoverable original test membership
16. TensorFlow, scikit-learn, or `tsfresh` version differences

Do not respond to a poor result by tuning against the target matrix before checking these items.

---

# Completion criteria

Do not declare the task complete merely because the script runs.

It is complete when:

- `data/` is used as the dataset root.
- All six classes are found.
- Ground truth comes from labels in paths or filenames.
- Existing transmission-detection code is used.
- Hopping starts at the first detected frequency bin.
- No hopping chunk exceeds 200 bins.
- Center cropping is absent from the Figure 11 path.
- One PSD row and one hopping chunk produce one prediction.
- Consecutive PSD rows are not used as LSTM timesteps.
- The 16 encoded feature coordinates remain the LSTM timesteps.
- The feature order matches the released scaler.
- `scaler.clip=True` has been removed.
- No entropy gate is used.
- No unknown class is used.
- No SNR gate is used.
- No classification width gate is used.
- No majority vote is used.
- No prediction is silently removed.
- The final matrix contains exactly 26,490 predictions.
- Each true-class row contains exactly 4,415 samples.
- Counts and normalized matrices are saved.
- The confusion-matrix image is saved.
- The exact commands and results are reported.

## Honest reporting requirement

The original public artifacts may not identify the exact 4,415 test samples per class used to evaluate the released checkpoint.

If the corrected pipeline is structurally faithful but does not reach exactly 94.25%, state that clearly.

Report:

- The obtained accuracy
- The obtained confusion matrix
- The deterministic split rule
- The seed used
- Candidate counts before balancing
- Skipped-file and skipped-row counts
- Any library-version differences
- Whether an original split manifest or seed was found

Do not describe a newly selected random subset as the authors’ exact test set unless there is evidence that it is the same subset.

Do not hardcode, manipulate, or selectively filter results to match the paper.

---

# Independent PyTorch Handoff

The current handoff pipeline does not run the released author code or load the
author's Keras checkpoints. It prepares data with `prepare_dataset.py` and
trains a new PyTorch autoencoder/LSTM with `train_pipeline.py`.

Start with these files:

- `AGENTS.md`: project context and prior findings.
- `data.md`: archive contents and corrected source-matrix interpretation.
- `mismatch_investigation.md`: why natural detector counts differ from the paper's balanced sample count.
- `segment_count_report.md`: counts at every detection and hopping stage.
- `prepare_dataset.py`: independent detector, hopping, feature artifacts, and split assignments.
- `train_pipeline.py`: PyTorch training and row-level evaluation.
- `count_segments.py`: fast count-only audit with no feature extraction or training.

## Dependencies

Install the project dependencies in an environment with a compatible Python
and PyTorch build:

```bash
python3 -m pip install -r requirements.txt
```

The archive is not committed to Git. Obtain `spectrum_bands.tar.gz` separately
or use an already extracted dataset directory.

## Count-Only Audit

Run this first. It reads the source matrices, estimates noise, detects
transmissions, and counts pre-hopping and post-hopping units. It does not run
`tsfresh`, PyTorch, training, or inference.

```bash
python3 count_segments.py \
  --data-dir data/opt/shared/alessio/Data_Transmitters_Identification/spectrum_bands_2 \
  --output-dir segment_counts
```

For an archive input:

```bash
python3 count_segments.py \
  --archive spectrum_bands.tar.gz \
  --output-dir segment_counts
```

The report writes `segment_count_report.json`,
`segment_count_by_class.csv`, and `segment_count_by_file.csv`.

## Prepare Features

The full preparation runs independent detection, sequential 215-bin hopping,
and 33-feature extraction. It writes `features.npy`, `segments.csv`,
`files.csv`, `rejections.csv`, `metadata.json`, and the feature-column list.
It can take hours on CPU because the existing `tsfresh` wrapper performs
repeated pandas row operations. Run the count-only audit first.

```bash
python3 prepare_dataset.py \
  --data-dir data/opt/shared/alessio/Data_Transmitters_Identification/spectrum_bands_2 \
  --output-dir prepared_dataset \
  --seed 42
```

The archive form is:

```bash
python3 prepare_dataset.py \
  --archive spectrum_bands.tar.gz \
  --output-dir prepared_dataset \
  --seed 42
```

For a fast end-to-end check, use the deterministic two-sensor, all-class
smoke subset:

```bash
python3 prepare_dataset.py \
  --data-dir data/opt/shared/alessio/Data_Transmitters_Identification/spectrum_bands_2 \
  --output-dir prepared_smoke \
  --smoke \
  --seed 42
```

## Train With A Segment Split

The prepared manifest contains a deterministic class-stratified `random_split`
with an approximately 80/20 allocation. Use:

```bash
python3 train_pipeline.py \
  --data-dir prepared_dataset \
  --output-dir training_class_random \
  --split class_random \
  --device auto
```

This split treats individual prepared samples as the unit. Different time rows
or hopping chunks from the same source file or detected transmission may land
in training and test. Exact byte-identical duplicate source files are kept in
one partition.

## Train With A Pre-Hopping Segment Split

Use `pre_hop_random` to stratify the detected-transmission time rows before
considering their hopping chunks:

```bash
python3 train_pipeline.py \
  --data-dir prepared_dataset \
  --output-dir training_pre_hop_random \
  --split pre_hop_random \
  --device auto
```

All hopping chunks sharing the same source file, detected transmission, and
time row remain in one partition. Adjacent time rows from the same transmission
can still cross the split.

## Train With A Sensor Split

The prepared manifest also contains a deterministic approximately 80/20
sensor-disjoint `sensor_split`. Use:

```bash
python3 train_pipeline.py \
  --data-dir prepared_dataset \
  --output-dir training_sensor \
  --split sensor \
  --device auto
```

Every sample from a sensor remains in one partition. The test sensor set never
appears in training. The preparation step links sensors with identical payloads
so confirmed duplicate data does not cross the split.

The trainer creates validation rows only from the prepared training partition.
It fits the `MinMaxScaler` on training rows only, trains the 33-to-16
autoencoder and PyTorch LSTM, and evaluates one prediction per test row without
transmission-level averaging.

For a two-epoch smoke training run:

```bash
python3 train_pipeline.py \
  --data-dir prepared_smoke \
  --output-dir training_smoke_pre_hop_random \
  --split pre_hop_random \
  --smoke \
  --device cpu

python3 train_pipeline.py \
  --data-dir prepared_smoke \
  --output-dir training_smoke_random \
  --split class_random \
  --smoke \
  --device cpu

python3 train_pipeline.py \
  --data-dir prepared_smoke \
  --output-dir training_smoke_sensor \
  --split sensor \
  --smoke \
  --device cpu
```

Each training output contains `metrics.json`, integer and normalized confusion
matrices, a classification report, test predictions, split indices, the
scaler, and a PyTorch `training_checkpoint.pt`.

## Verification

Run the focused tests before committing changes:

```bash
python3 -m pytest -q
python3 -m py_compile prepare_dataset.py count_segments.py train_pipeline.py
```

The public artifacts do not expose the authors' exact test membership, random
seed, or complete strategy-specific sample construction. Any 22,415-per-class
selection is a deterministic reconstruction, not a claim of exact test-set
identity.

---

# Final response format

When finished, provide:

## Files changed

List each changed file and the reason.

## Commands run

Include the exact smoke-run and full-run commands.

## Dataset/preprocessing counts

Report:

- Files per class
- Detected transmissions per class
- Hopping chunks per class
- Candidate PSD segments per class
- Final test samples per class
- Skipped files or segments and reasons

## Result

Report:

- Overall accuracy
- Integer confusion matrix
- Row-normalized confusion matrix
- Per-class accuracy
- Difference from the reported 94.25%
- Output file paths

## Remaining limitation

State whether the exact original test membership was recoverable from the public artifacts.
