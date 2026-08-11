# Dataset Analysis: Wireless Technology Classification

**Paper:** "A Framework for Wireless Technology Classification using Crowdsensing Platforms"
**Analyzed source:** `spectrum_bands.tar.gz`

## Source Matrices

Each `.npy` file stores a two-dimensional PSD matrix with shape `(time, frequency)`. The first dimension stores PSD rows; the second stores frequency bins.

| Metric | Value |
|---|---:|
| Source matrices | 232 |
| Published sensor directories represented | 40 |
| Stored rows | 99,606 |
| Usable rows (`frequency_bins > 0`) | 98,262 |
| Summed frequency-bin widths | 745,884 |
| Frequency-empty arrays | 3 |
| Valid filename frequency pairs | 231 / 232 |

Technology labels come from the technology token in each filename. Malformed frequency metadata does not remove a file from its technology count. A frequency pair is nullable when the filename has no valid increasing pair after that token.

## Technology Counts

| Technology | Files | Stored rows | Usable rows | Summed frequency bins |
|---|---:|---:|---:|---:|
| DAB | 32 | 13,778 | 13,778 | 233,026 |
| DVB-T | 34 | 14,797 | 14,347 | 178,985 |
| FM | 41 | 17,622 | 17,622 | 108,024 |
| GSM | 46 | 19,413 | 18,964 | 43,155 |
| LTE | 41 | 17,616 | 17,171 | 137,318 |
| TETRA | 38 | 16,380 | 16,380 | 45,376 |

## Three Count Levels

- **Source matrix:** one published `.npy` file containing time rows and frequency bins.
- **Detected transmission:** a region found by the transmission detector inside a source matrix; its count depends on detector behavior.
- **Classifier sample:** a feature vector made from a PSD row and, where applicable, a hopping chunk of a detected transmission.

The 98,262 usable source rows are not detected-transmission or classifier-sample counts. Detection can produce zero or multiple transmissions per matrix, and hopping can produce multiple classifier samples from one row.

## Paper Reference

The paper reports 108,000 training PSD segments and 26,490 testing PSD segments. Those are paper experiment counts, not counts inferred from the 232 source matrices. The paper also reports 282 sensor-hours; this report repeats that published figure and does not estimate duration from rows, frequency widths, or hopping chunks.

## Reproduction Pipeline

`prepare_dataset.py` implements the independent processing path used by this reproduction:

1. Estimate a sensor noise floor from the quietest available 215-bin block.
2. Detect occupied transmission boundaries with fixed, class-independent parameters.
3. Preserve transmissions up to 215 bins and split wider transmissions into sequential, non-overlapping 215-bin chunks from the first detected bin.
4. Extract one 33-feature classifier sample for each finite time-row/chunk pair.
5. Store a class-stratified segment-level 80/20 split and a sensor-disjoint grouped 80/20 split.

The detector and hopping implementation does not import or execute the released author code. Technology labels provide ground truth only; detection does not use class-specific width, frequency, SNR, or entropy gates.

Run `python3 prepare_dataset.py --archive spectrum_bands.tar.gz --output-dir prepared_dataset`. The natural population can contain millions of candidates because the paper selected 134,490 balanced samples rather than using every valid detected row/chunk pair.

## Frequency-Empty Arrays

- `SpectrumBands_9350_941_gsm_slv_935_491.npy` (`449 x 0`)
- `SpectrumBands_484_419_dvbt_cz_484_519.npy` (`450 x 0`)
- `SpectrumBands_770_80_lte_Jap_770_803.npy` (`445 x 0`)

## Duplicate Sequence Analysis

### Identical File Pairs (Exact Byte-for-Byte Duplicates)

| Group | Files | Sensors | Shape | Description |
|-------|-------|---------|-------|-------------|
| 1 | 3 | Britof, HuldOne, NFM-Electrosense-01 | Various (0 cols) | Empty files with identical hash |
| 2 | 2 | Defcon1, EDLV | (426, 3226) | **LTE files are IDENTICAL** |
| 3 | 2 | Skap_French_Riviera, imdea_adsb | (451, 3226) | **LTE files are IDENTICAL** |
| 4 | 2 | oha_sense1 (Sep_1, Sep_5) | (451, 3226) | **DAB files are IDENTICAL** |

**Impact:** 4 files are exact duplicates of other files. These should be deduplicated before training.

### Cross-Sensor Same-Filename Files

Many files share the same filename across different sensors (capturing the same frequency band). These are **NOT duplicates** - they contain different data from different locations:

| Filename | Copies | Sensors | Different Data? |
|----------|--------|---------|-----------------|
| `SpectrumBands_791_821_lte_Esp_791_821.npy` | 6 | Skap, URJC1, bcn-L, donostia, imdea, miguel | Yes (different shapes/means) |
| `SpectrumBands_734_768_lte_usa_734_768.npy` | 3 | Bandon_Oregon, Oreland, Princeton1 | Yes |
| `SpectrumBands_86_110_fm_usa_86_110.npy` | 3 | Bandon_Oregon, Oreland, Princeton1 | Yes |
| `SpectrumBands_791_821_lte_Swis_791_821.npy` | 3 | Geneva, Sensorix, dipolkurz | Yes |
| `SpectrumBands_924_929_gsm_Esp_924_929.npy` | 3 | bcn-L, imdea, miguel | Yes |

**Exception:** Skap_French_Riviera and imdea_adsb have **identical** LTE data (same file copied to two sensors).

### Same-Sensor Multi-Date Files

Some sensors have data from multiple dates. These are generally **different** (collected at different times):

| Sensor | Files | Same Date? | Identical? |
|--------|-------|------------|------------|
| oha_sense1 | DAB (Sep_1, Sep_5) | Different dates | **IDENTICAL** (bug) |
| oha_sense1 | DVB-T (Sep_1, Sep_5) | Different dates | Different (shape/mean differ) |
| oha_sense1 | LTE (Sep_1, Sep_5) | Different dates | Different |
| scalessio | All 6 techs (May_2, May_3) | Different dates | Different |
| leganes_rack_3 | GSM/DVB-T (May_1, May_2) | Different dates | Different |
| alcorcon1 | All techs (Feb_1, Feb_2, Feb_3) | Different dates | Different |

### Within-File Duplicate Sequences

**Result:** No duplicate columns (time segments) found within any file. Every PSD measurement in every file is unique.

### Near-Duplicate Sequences (Correlation > 0.999)

**Result:** No near-duplicate sequences found. All time segments within files are sufficiently different.

### Summary of Duplicates

| Type | Count | Impact |
|------|-------|--------|
| Exact file duplicates | 4 files | Remove before training |
| Identical cross-sensor files | 1 pair (Skap/imdea) | Remove one |
| Identical multi-date files | 1 pair (oha_sense1 DAB) | Remove one |
| **Total unique files to remove** | **5 files** | Reduces dataset from 232 to 227 unique files |

---

## Analysis Script

Run `analyze_dataset.py --archive spectrum_bands.tar.gz` for the archive or `analyze_dataset.py --data-dir <directory>` for an extracted tree.
