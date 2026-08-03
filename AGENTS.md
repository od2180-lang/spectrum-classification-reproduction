# Agent Instructions

1. Code for this project will go in the repository at https://github.com/od2180-lang/spectrum-classification-reproduction

2. Large data files should go in remote object storage. Credentials for object storage are in the `.env` file.

3. If not logged into GitHub (`gh auth status`), instruct the user to run `gh auth login` and follow the browser-based authentication flow.

---

## Latest Experiment: Authors' Exact Pipeline (replicate_figure11.py)

Uses authors' exact `TechClass.py` with 3 minimal patches: dead import removed, model path fixed, lr deprecation fixed.

### Results on our 2490 transmissions

| Metric | Value |
|--------|-------|
| Overall accuracy (entropy gate) | **68.42%** (221/323) |
| Unknown filtered by entropy gate | 2167/2490 (87%) |
| Paper's reported accuracy | **94.25%** |

### Per-Class Accuracy

| Technology | Correct | Total | Accuracy |
|------------|---------|-------|----------|
| DAB | 217 | 217 | 100.00% |
| DVB-T | 0 | 1 | 0.00% |
| FM | 4 | 6 | 66.67% |
| GSM | 0 | 0 | - |
| LTE | 0 | 0 | - |
| TETRA | 0 | 99 | 0.00% |

### Bugs in Authors' Code (Gap to Paper)

1. **Buggy frequency ID** (TechClass.py:269-283): DVB-T/GSM/LTE always become `'unkn'` (`*1e6` on both sides of comparison)
2. **`list_entropy` accumulation**: never cleared between files, compounds across entire run
3. **`h + threshold_alpha`** on low-SNR path: inflates accumulated entropy further
4. **LSTM bias toward DAB**: 217/323 confident predictions are DAB

### Key Pipeline Details

- **Width gates** (TechClass.py:285-312): checks ORIGINAL width BEFORE cropping
  - DAB: 120-240, DVB-T: ≥400, FM: none, GSM: 14-35, LTE: >700, TETRA: <10
- **Raw transmissions** in `detected_transmissions/` (with CSV metadata for SNR)
- **Entropy gate**: threshold 0.7, majority vote on 50 segment predictions

