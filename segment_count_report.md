# Segment Count Report

This report counts the data units produced by the independent detection and
hopping pipeline. It does not run feature extraction, model training, or
classification.

Source data:

```text
data/opt/shared/alessio/Data_Transmitters_Identification/spectrum_bands_2
```

The audit can be regenerated with:

```bash
python3 count_segments.py \
  --data-dir data/opt/shared/alessio/Data_Transmitters_Identification/spectrum_bands_2 \
  --output-dir segment_counts
```

## Unit Definitions

| Unit | Definition |
|---|---|
| Source matrix | One published `.npy` file, stored as `(time rows, frequency bins)` |
| Coarse interval | One contiguous frequency interval above the 5 dB occupancy threshold |
| Detected transmission | One final interval after CV and 3 dB edge processing |
| Pre-hopping row | One time row from one detected transmission |
| Hopping chunk | One sequential 215-bin chunk from a detected transmission; narrow transmissions remain one chunk |
| Post-hopping candidate | One time row from one hopping chunk |

The report shows both a 360-row count and an all-available-row count. The
paper describes TDS processing `K = 360` consecutive time periods, while the
published matrices often contain more than 360 rows.

## Overall Counts

| Pipeline stage | Count |
|---|---:|
| Source matrices | 232 |
| Valid non-empty matrices | 229 |
| Published sensor directories | 40 |
| Usable source rows | 98,262 |
| Source rows within the 360-row TDS window | 80,873 |
| Coarse occupancy intervals | 6,691 |
| Final detected transmissions | 6,999 |
| Pre-hopping rows, 360-row interpretation | 2,466,725 |
| Pre-hopping rows, all available rows | 3,017,154 |
| Hopping chunks | 7,444 |
| Post-hopping candidates, 360-row interpretation | 2,623,298 |
| Post-hopping candidates, all available rows | 3,208,990 |
| Post-hopping candidates excluding width-1 chunks, 360-row interpretation | 2,101,786 |
| Post-hopping candidates excluding width-1 chunks, all available rows | 2,570,842 |

The paper reports 108,000 training and 26,490 testing samples, or 134,490
total samples. The current natural post-hopping population is therefore:

```text
3,208,990 / 134,490 = 23.86 times the paper reference
```

The count-only pipeline does not claim that 134,490 is a natural detector
output. It is the paper's balanced experiment population.

## Counts By Technology

Counts use the 360-row interpretation for row totals.

| Technology | Files | Coarse intervals | Detections | Pre-hop rows | Hopping chunks | Post-hop candidates |
|---|---:|---:|---:|---:|---:|---:|
| DAB | 32 | 752 | 752 | 268,844 | 801 | 285,071 |
| DVB-T | 34 | 1,590 | 1,590 | 549,202 | 1,679 | 581,008 |
| FM | 41 | 2,537 | 2,832 | 1,009,276 | 2,857 | 1,018,208 |
| GSM | 46 | 501 | 511 | 174,218 | 514 | 175,088 |
| LTE | 41 | 719 | 719 | 255,727 | 998 | 354,465 |
| TETRA | 38 | 592 | 595 | 209,458 | 595 | 209,458 |
| **Total** | **232** | **6,691** | **6,999** | **2,466,725** | **7,444** | **2,623,298** |

The corresponding all-row post-hopping counts are DAB 355,194, DVB-T 717,317,
FM 1,243,184, GSM 213,857, LTE 422,409, and TETRA 257,029.

## Feature Stage

The full feature-generation pass has not been completed. A previous attempt
exceeded 30 minutes because the existing `tsfresh` wrapper computes 33 features
through repeated pandas row operations for every hopping chunk.

The smoke pipeline did complete feature extraction on a controlled subset:

```text
720 feature rows
33 features per row
120 rows per class
```

This confirms that the feature stage consumes post-hopping candidates, but it
does not provide a full-archive count of finite feature rows. Width-1 chunks
are known to produce undefined derivative features and are excluded from the
width-greater-than-one counts above.

## Paper Comparison

The paper does not publish the number of coarse intervals, detections, hopping
chunks, or strategy-specific feature vectors. It reports only the balanced
training and testing populations. Its Figure 10 compares baseline, Strategy I,
Strategy II, and hopping feature preparation, but gives no per-strategy sample
counts.

The current audit therefore separates the natural pipeline population from
the paper reference. Selecting 22,415 samples per class would create a
paper-sized reconstruction, but it would be a declared balancing step rather
than evidence that the detector naturally produced 134,490 samples.

## Output Files

The machine-readable audit is stored in:

- `segment_counts/segment_count_report.json`
- `segment_counts/segment_count_by_class.csv`
- `segment_counts/segment_count_by_file.csv`
