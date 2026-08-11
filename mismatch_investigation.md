# Paper Count Mismatch

## Paper Evidence

The paper reports 108,000 training samples and 26,490 testing samples in
Table II, for 134,490 total samples. It says that class instances are equally
balanced, which implies 22,415 samples per class. The paper does not report the
number of coarse detections, final detections, hopping chunks, or feature
vectors produced by each strategy in Figure 10.

The paper defines a PSD segment as one frequency vector from one measurement.
The TDS first detects transmission boundaries in a matrix of consecutive
measurements. Section VII-B-3 then applies baseline, Strategy I, Strategy II,
or hopping feature preparation to the same type of PSD segment. The paper does
not state whether the reported 134,490 count is before or after a strategy
expands one segment into multiple feature vectors.

## Independent Count Audit

The audit uses `prepare_dataset.py` on the published, already filtered
`spectrum_bands` matrices. It does not use the released author implementation.

| Stage | Count |
|---|---:|
| Source matrices | 232 |
| Valid matrices | 229 |
| Files with detections | 223 |
| Coarse occupancy intervals | 6,691 |
| Final detections | 6,999 |
| Pre-hopping rows, 360-row interpretation | 2,466,725 |
| Pre-hopping rows, all available rows | 3,017,154 |
| Hopping chunks | 7,444 |
| Post-hopping rows, 360-row interpretation | 2,623,298 |
| Post-hopping rows, all available rows | 3,208,990 |
| Paper reference total | 134,490 |

Excluding one-bin detections still leaves 1,945,213 pre-hopping rows under the
360-row interpretation. The count is therefore not explained by one-bin
noise fragments alone.

## Controlled Variants

| Variant | Final detections | Post-hopping rows, all available rows |
|---|---:|---:|
| Site-level quietest-block noise | 6,999 | 3,208,990 |
| File-local quietest-block noise | 6,760 | 3,021,748 |
| Current `tds.py` noise calculation | 4,741 | 2,075,589 |
| 30th-percentile dB noise | 7,271 | 3,186,687 |
| CV splitting disabled | 6,691 | 3,136,065 |
| Reference `ChannelDetector` | 2,231 | 1,164,778 |

The detector choices affect the result, but no tested paper-like variant
produces approximately 134,490 samples.

## Root Cause

The archive README says the published files are specific labelled frequency
portions extracted from full-spectrum scans. The paper's TDS description,
however, operates on full-spectrum scan data and derives a noise floor from
the full scanned spectrum. We no longer have the original full-sweep matrices
or the authors' intermediate detected-transmission files.

Our current audit applies TDS independently to each labelled band. Each
detected frequency region then contributes every available time row. A single
source file can therefore produce many detections multiplied by roughly 360 to
450 rows. Sequential hopping adds further row/chunk candidates.

The 134,490 number is the paper's final balanced train/test population, not a
published natural detector-output count. The paper does not provide enough
information to prove that the authors retained every row/chunk candidate or to
reconstruct their filtering and selection from this archive.

## Consequence

Do not tune detector thresholds until the count reaches 134,490. That would
turn the reported model population into a hidden calibration target. Report
the natural candidate counts separately, then use the paper's 22,415-per-class
population only as a declared sample-selection reconstruction if a comparable
classifier evaluation is required.

The raw audit files are in `segment_counts/`:

- `segment_count_report.json`
- `segment_count_by_class.csv`
- `segment_count_by_file.csv`
