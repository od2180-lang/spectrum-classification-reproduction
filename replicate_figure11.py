#!/usr/bin/env python3
"""
Replicate Figure 11: Authors' exact pipeline with raw transmissions
Uses detected_transmissions/ (raw, uncropped) with CSV metadata for SNR.
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tsfresh.feature_extraction.feature_calculators import (
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
)

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, 'figures')
DATA_DIR = os.path.join(BASE, 'detected_transmissions')
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_BASE = os.path.join(BASE, 'PSD-technology-classification-framework/TCpackage/resources')
SCALER_PATH = os.path.join(MODEL_BASE, 'scaler/_AE16_LSTM_Scaler_.save')
ENCODER_PATH = os.path.join(MODEL_BASE, 'save-DL-models/Autoencoder_DNN/TrainAllSensorHop_16_feat_mse_relu/saved-model-49-0.0002.hdf5')
LSTM_PATH = os.path.join(MODEL_BASE, 'save-DL-models/LSTM_TrainWithAE/TrainAllSensorHop_DNNAE16_LSTM_mse_relu/saved-model-110-0.97.hdf5')

LABELS = ['dab', 'dvbt', 'fm', 'gsm', 'lte', 'tetra']
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}
N_CLASSES = 6
list_entropy = []  # Bug: never cleared between files (matches TechClass.py:33)

FEATURE_COLS = [
    'abs_energy', 'absolute_sum_of_changes', 'benford_correlation', 'cid_ce',
    'count_above_mean', 'count_below_mean', 'first_location_of_maximum',
    'first_location_of_minimum', 'has_duplicate', 'has_duplicate_max',
    'has_duplicate_min', 'kurtosis', 'last_location_of_maximum',
    'last_location_of_minimum', 'longest_strike_above_mean',
    'longest_strike_below_mean', 'maximum', 'mean', 'mean_abs_change',
    'mean_change', 'mean_second_derivative_central', 'median', 'minimum',
    'number_cwt_peaks', 'number_peaks', 'quantile', 'root_mean_square',
    'skewness', 'standard_deviation', 'sum_of_reoccurring_values',
    'sum_values', 'variance', 'variation_coefficient'
]


def load_encoder():
    encoder = Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=[33]),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu')
    ])
    decoder = Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=[16]),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(33)
    ])
    autoencoder = Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.load_weights(ENCODER_PATH)
    return encoder


def load_lstm():
    model = Sequential([
        tf.keras.layers.LSTM(32, activation='relu', input_shape=(16, 1), return_sequences=True),
        tf.keras.layers.LSTM(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='softmax'),
        tf.keras.layers.Dropout(0.001),
        tf.keras.layers.Dense(N_CLASSES, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    model.load_weights(LSTM_PATH)
    return model


def extract_features(dta):
    features = []
    for row in dta:
        features.append([
            abs_energy(row), absolute_sum_of_changes(row),
            benford_correlation(row), cid_ce(row, True),
            count_above_mean(row), count_below_mean(row),
            first_location_of_maximum(row), first_location_of_minimum(row),
            has_duplicate(row), has_duplicate_max(row), has_duplicate_min(row),
            kurtosis(row), last_location_of_maximum(row),
            last_location_of_minimum(row), longest_strike_above_mean(row),
            longest_strike_below_mean(row), maximum(row), mean(row),
            mean_abs_change(row), mean_change(row),
            mean_second_derivative_central(row), median(row), minimum(row),
            number_cwt_peaks(row, n=3), number_peaks(row, n=3),
            quantile(row, 0.5), root_mean_square(row), skewness(row),
            standard_deviation(row), sum_of_reoccurring_values(row),
            sum_values(row), variance(row), variation_coefficient(row)
        ])
    return np.array(features)


def refine_df(df):
    df.replace([np.inf], np.nan, inplace=True)
    df['skewness'] = df['skewness'].fillna(0)
    df['mean_second_derivative_central'] = df['mean_second_derivative_central'].fillna(0)
    df['kurtosis'] = df['kurtosis'].fillna(0)
    return df


def extract2MHz(dta):
    center = round(dta.shape[1] / 2)
    return dta[:, (center - 100):(center + 100)]


def check_width_gate(tech, width):
    if tech == 'dab' and 120 <= width <= 240:
        return True
    elif tech == 'dvbt' and width >= 400:
        return True
    elif tech == 'fm':
        return True
    elif tech == 'gsm' and 14 <= width <= 35:
        return True
    elif tech == 'lte' and width > 700:
        return True
    elif tech == 'tetra' and width < 10:
        return True
    return False


def calc_entropy(probs):
    """Shannon entropy (base 2) of a single segment's 6-class probability vector."""
    ent = 0.0
    for p in probs:
        if p > 0:
            ent += p * math.log(p, 2)
    return -ent


def score_entropy_pred(file_probs, threshold=0.7):
    """
    Replicates TechClass.py:scoreEntropyPred (lines 155-160).
    Computes entropy per-segment, appends to global list_entropy,
    then averages the accumulated list.
    Bug: list_entropy is never cleared between files.
    """
    global list_entropy
    for elem in file_probs:
        h = calc_entropy(elem)
        list_entropy.append(h)
    entropy_avg = np.mean(list_entropy)
    if entropy_avg > threshold:
        return 6, entropy_avg
    prob_mean = np.mean(file_probs, axis=0)
    pred = np.argmax(prob_mean)
    return pred, entropy_avg


def main():
    print("=" * 60)
    print("  Replicate Figure 11: Authors' Exact Pipeline (Raw TX)")
    print("=" * 60)

    print("\n[1/3] Loading authors' pre-trained models...")
    scaler = joblib.load(SCALER_PATH)
    encoder = load_encoder()
    lstm_model = load_lstm()
    print("  Done.")

    print("\n[2/3] Loading and classifying raw transmissions...")
    npy_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.npy')])
    print(f"  Total raw transmissions: {len(npy_files)}")

    results = []
    skipped_width = 0
    skipped_tech = 0
    classified = 0

    for fi, fname in enumerate(npy_files):
        # Parse technology from filename
        tech = None
        for t in LABELS:
            if t in fname.lower():
                tech = t
                break
        if tech is None:
            skipped_tech += 1
            continue

        # Load raw transmission
        dta = np.load(os.path.join(DATA_DIR, fname), allow_pickle=True)
        if dta.ndim == 1:
            dta = dta.reshape(1, -1)
        width = dta.shape[1]

        # Width gate (TechClass.py:285-312)
        if not check_width_gate(tech, width):
            skipped_width += 1
            continue

        # Read SNR from CSV (TechClass.py:262-263)
        csv_path = os.path.join(DATA_DIR, fname.replace('.npy', '.csv'))
        if os.path.exists(csv_path):
            meta = pd.read_csv(csv_path)
            snr = meta['SNR'].values[0]
        else:
            snr = 10.0  # default high SNR

        # Trim to 50 time segments (TechClass.py:227)
        dta = dta[:50, :]

        # Crop to center 200 bins if wide enough (TechClass.py:228-229)
        if dta.shape[1] >= 200:
            dta = extract2MHz(dta)

        # Extract 33 features (TechClass.py:230)
        features = extract_features(dta)

        # Clean NaN/inf (TechClass.py:140-146)
        df = pd.DataFrame(features, columns=FEATURE_COLS)
        df = refine_df(df)
        X = df.values

        # Scale (TechClass.py:202)
        X_scaled = scaler.transform(X)

        # Encode (TechClass.py:203)
        X_encoded = encoder.predict(X_scaled, verbose=0)

        # Reshape for LSTM (TechClass.py:204)
        X_input = X_encoded.reshape(50, 16, 1)

        # Classify (TechClass.py:206)
        probs = lstm_model.predict(X_input, verbose=0)

        # SNR gate (TechClass.py:199-222)
        if snr <= 3:
            # Low SNR → label unknown
            label = 6
            entropy = 0.0
        else:
            # Normal: entropy gate + majority vote
            label, entropy = score_entropy_pred(probs)

        # No-gate prediction: majority vote without entropy gate
        prob_mean = np.mean(probs, axis=0)
        no_gate_pred = int(np.argmax(prob_mean))

        results.append({
            'file': fname,
            'true_tech': tech,
            'pred_label': label,
            'pred_tech': LABELS[label] if label < 6 else 'unkn',
            'no_gate_label': no_gate_pred,
            'entropy': entropy,
            'width': width,
            'snr': snr,
            'correct': LABELS[label] == tech if label < 6 else False
        })

        classified += 1
        if (fi + 1) % 200 == 0:
            print(f"  Processed {fi + 1}/{len(npy_files)} files")

    print(f"\n  Classified: {classified}")
    print(f"  Skipped (width gate): {skipped_width}")
    print(f"  Skipped (unknown tech): {skipped_tech}")

    # Evaluate
    print("\n[3/3] Evaluating...")
    df_results = pd.DataFrame(results)

    # WITH entropy gate
    known_mask = df_results['pred_label'] != 6
    df_known = df_results[known_mask]

    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for _, row in df_known.iterrows():
        true_idx = LABEL_TO_IDX[row['true_tech']]
        pred_idx = row['pred_label']
        cm[true_idx, pred_idx] += 1

    cm_norm = cm.astype(float)
    for i in range(N_CLASSES):
        s = cm_norm[i].sum()
        if s > 0:
            cm_norm[i] /= s

    acc = np.trace(cm) / cm.sum() * 100 if cm.sum() > 0 else 0
    unknown_count = (df_results['pred_label'] == 6).sum()

    print(f"\n=== WITH ENTROPY GATE (threshold=0.7) ===")
    print(f"  Total classified: {len(df_results)}")
    print(f"  Unknown (filtered): {unknown_count}")
    print(f"  Known predictions: {len(df_known)}")
    print(f"  Overall Accuracy: {acc:.2f}% ({np.trace(cm)}/{cm.sum()})")

    header = "        " + "  ".join(f"{l:>6s}" for l in [x.upper() for x in LABELS])
    print("\n=== CONFUSION MATRIX (normalized) ===")
    print(header)
    for i, row in enumerate(cm_norm):
        vals = "  ".join(f"{v:6.2f}" for v in row)
        print(f"  {LABELS[i].upper():>6s}: {vals}")

    print("\n=== PER-CLASS ACCURACY ===")
    for i in range(N_CLASSES):
        cls_count = cm[i].sum()
        cls_correct = cm[i, i]
        cls_acc = cls_correct / cls_count * 100 if cls_count > 0 else 0
        print(f"  {LABELS[i].upper():>6s}: {cls_acc:.2f}% ({cls_correct}/{cls_count})")

    # WITHOUT entropy gate
    print(f"\n=== NO ENTROPY GATE ===")
    cm_full = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for _, row in df_results.iterrows():
        true_idx = LABEL_TO_IDX[row['true_tech']]
        pred_idx = row['no_gate_label']
        cm_full[true_idx, pred_idx] += 1

    cm_full_norm = cm_full.astype(float)
    for i in range(N_CLASSES):
        s = cm_full_norm[i].sum()
        if s > 0:
            cm_full_norm[i] /= s

    acc_full = np.trace(cm_full) / cm_full.sum() * 100
    print(f"  Total classified: {len(df_results)}")
    print(f"  Overall Accuracy: {acc_full:.2f}% ({np.trace(cm_full)}/{cm_full.sum()})")

    print("\n=== CONFUSION MATRIX (normalized, no gate) ===")
    print(header)
    for i, row in enumerate(cm_full_norm):
        vals = "  ".join(f"{v:6.2f}" for v in row)
        print(f"  {LABELS[i].upper():>6s}: {vals}")

    print("\n=== PER-CLASS ACCURACY (no gate) ===")
    for i in range(N_CLASSES):
        cls_count = cm_full[i].sum()
        cls_correct = cm_full[i, i]
        cls_acc = cls_correct / cls_count * 100 if cls_count > 0 else 0
        print(f"  {LABELS[i].upper():>6s}: {cls_acc:.2f}% ({cls_correct}/{cls_count})")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, cm_n, title, acc_v in [(ax1, cm_norm, 'With Entropy Gate', acc),
                                    (ax2, cm_full_norm, 'No Entropy Gate', acc_full)]:
        im = ax.imshow(cm_n, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        ax.figure.colorbar(im, ax=ax)
        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels([l.upper() for l in LABELS], fontsize=10)
        ax.set_yticklabels([l.upper() for l in LABELS], fontsize=10)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i, f'{cm_n[i,j]:.2f}', ha='center', va='center',
                        color='white' if cm_n[i,j] > 0.5 else 'black', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        ax.set_ylabel('True', fontsize=11, fontweight='bold')
        ax.set_title(f'{title}\nAccuracy: {acc_v:.1f}%', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, 'figure11_lstm_confusion.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    print("Done!")


if __name__ == '__main__':
    main()
