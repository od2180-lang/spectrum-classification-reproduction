import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report

from tds import TransmissionDetectionSystem
from feature_extraction import FeatureExtractor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================
# Constants
# ============================================================
TECHNOLOGIES = ['dab', 'dvbt', 'fm', 'gsm', 'lte', 'tetra']
TECH_MAP = {'dab': 'DAB', 'dvbt': 'DVB-T', 'fm': 'FM', 'gsm': 'GSM', 'lte': 'LTE', 'tetra': 'TETRA'}
CLASS_NAMES = ['DAB', 'DVB-T', 'FM', 'GSM', 'LTE', 'TETRA']
DATASET_PATH = 'dataset/opt/shared/alessio/Data_Transmitters_Identification/spectrum_bands_2'
CHUNK_SIZE = 215
MAX_TIME_SWEEPS = 50

# Paper hyperparameters (Section VII)
AE_EPOCHS = 550
AE_PATIENCE = 10
AE_LR = 0.001
LSTM_EPOCHS = 550
LSTM_PATIENCE = 10
LSTM_LR = 0.001
DROPOUT_RATE = 0.001
TEST_SIZE = 0.2


def get_label(filename):
    for tech in TECHNOLOGIES:
        if tech in filename.lower():
            return TECH_MAP[tech]
    return None


# ============================================================
# Step 1: Run TDS + Feature Extraction on all data
# ============================================================
def extract_all_data():
    """
    For each detected transmission:
    - Extract features from all K time sweeps (each sweep -> 32 features)
    - Store as (feature_matrix, label) where feature_matrix is (K, 32)
    """
    tds = TransmissionDetectionSystem()
    extractor = FeatureExtractor()

    all_features = []  # list of (K_i, 32) arrays
    all_labels = []

    for root, dirs, files in os.walk(DATASET_PATH):
        for f in sorted(files):
            if not f.endswith('.npy'):
                continue
            label = get_label(f)
            if label is None:
                continue

            data = np.load(os.path.join(root, f), allow_pickle=True)
            transmissions = tds.detect_transmissions(data)

            for start, end in transmissions:
                tx_data = data[:, start:end + 1]
                width = tx_data.shape[1]

                # Pad/truncate to 215 bins (2 MHz)
                if width < CHUNK_SIZE:
                    padded = np.zeros((tx_data.shape[0], CHUNK_SIZE))
                    padded[:, :width] = tx_data
                    tx_data = padded
                else:
                    tx_data = tx_data[:, :CHUNK_SIZE]

                # Extract features from each time sweep
                num_sweeps = min(tx_data.shape[0], MAX_TIME_SWEEPS)
                feat_list = []
                for k in range(num_sweeps):
                    feats = extractor.extract_features(tx_data[k, :])
                    if np.all(np.isfinite(feats)):
                        feat_list.append(feats)

                if len(feat_list) >= 5:
                    all_features.append(np.array(feat_list))
                    all_labels.append(label)

    return all_features, np.array(all_labels)


# ============================================================
# Step 2: Train AutoEncoder (Table Ia)
# ============================================================
def build_autoencoder():
    """
    Paper Table Ia:
    Input(32) -> Dense(64,relu) -> Dense(32,relu) -> Dense(16,relu)
    -> Dense(32,relu) -> Dense(64,relu) -> Dense(32,linear)
    """
    inp = keras.Input(shape=(32,))
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.Dense(32, activation='relu')(x)
    encoded = layers.Dense(16, activation='relu', name='compressed')(x)
    x = layers.Dense(32, activation='relu')(encoded)
    x = layers.Dense(64, activation='relu')(x)
    decoded = layers.Dense(32, activation='linear')(x)

    autoencoder = keras.Model(inp, decoded)
    encoder = keras.Model(inp, encoded)
    autoencoder.compile(optimizer=keras.optimizers.Adam(AE_LR), loss='mse')
    return autoencoder, encoder


def train_autoencoder(all_features):
    """Train AE on all individual feature vectors (flatten all time sweeps)."""
    print("=" * 60)
    print("TRAINING AUTOENCODER")
    print("=" * 60)

    # Flatten: each time sweep becomes one sample
    flat = np.concatenate([f for f in all_features], axis=0)
    print(f"Total feature vectors: {flat.shape[0]}, dim: {flat.shape[1]}")

    # Shuffle and scale
    flat = shuffle(flat, random_state=42)
    scaler = StandardScaler()
    flat_scaled = scaler.fit_transform(flat)

    # Split
    X_train, X_val = train_test_split(flat_scaled, test_size=0.1, random_state=42)

    # Build and train
    autoencoder, encoder = build_autoencoder()
    print("\nAutoEncoder architecture:")
    autoencoder.summary()

    callbacks = [keras.callbacks.EarlyStopping(patience=AE_PATIENCE, restore_best_weights=True)]
    history = autoencoder.fit(
        X_train, X_train,
        epochs=AE_EPOCHS, batch_size=32,
        validation_data=(X_val, X_val),
        callbacks=callbacks, verbose=1
    )

    # Save
    autoencoder.save('autoencoder_model.keras')
    encoder.save('encoder_model.keras')
    joblib.dump(scaler, 'scaler.pkl')

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.title('AutoEncoder Training'); plt.legend()
    plt.tight_layout(); plt.savefig('ae_loss.png', dpi=150); plt.close()

    print(f"\nAE saved. Train loss: {history.history['loss'][-1]:.6f}, Val loss: {history.history['val_loss'][-1]:.6f}")
    return encoder, scaler


# ============================================================
# Step 3: Train LSTM (Table Ib)
# ============================================================
def build_lstm():
    """
    Paper Table Ib:
    Input(16) -> LSTM(32) -> LSTM(16) -> Dense(16,relu) -> Dropout -> Dense(6,softmax)
    """
    model = keras.Sequential([
        layers.Input(shape=(None, 16)),
        layers.LSTM(32, return_sequences=True, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.LSTM(16, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu'),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(6, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(LSTM_LR),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def prepare_lstm_data(all_features, all_labels, encoder, scaler):
    """Encode each transmission's feature sequence and pad."""
    le = LabelEncoder()
    y = le.fit_transform(all_labels)

    # Encode and collect
    sequences = []
    for feat_matrix in all_features:
        scaled = scaler.transform(feat_matrix)
        encoded = encoder.predict(scaled, verbose=0)
        sequences.append(encoded)

    # Pad to same length
    max_len = max(s.shape[0] for s in sequences)
    X = np.zeros((len(sequences), max_len, 16))
    lengths = []
    for i, seq in enumerate(sequences):
        length = seq.shape[0]
        X[i, :length, :] = seq
        lengths.append(length)

    return X, y, np.array(lengths), le


def train_lstm(all_features, all_labels):
    print("\n" + "=" * 60)
    print("TRAINING LSTM CLASSIFIER")
    print("=" * 60)

    # Load encoder and scaler
    encoder = keras.models.load_model('encoder_model.keras')
    scaler = joblib.load('scaler.pkl')

    # Prepare data
    X, y, lengths, le = prepare_lstm_data(all_features, all_labels, encoder, scaler)
    print(f"Sequences: {X.shape}, Labels: {len(np.unique(y))} classes")
    for tech in CLASS_NAMES:
        if tech in le.classes_:
            print(f"  {tech}: {np.sum(y == le.transform([tech])[0])}")
        else:
            print(f"  {tech}: 0")

    # Split
    X_train, X_test, y_train, y_test, len_train, len_test = train_test_split(
        X, y, lengths, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Build and train
    model = build_lstm()
    print("\nLSTM architecture:")
    model.summary()

    callbacks = [keras.callbacks.EarlyStopping(patience=LSTM_PATIENCE, restore_best_weights=True)]
    history = model.fit(
        X_train, y_train,
        epochs=LSTM_EPOCHS, batch_size=32,
        validation_split=0.1,
        callbacks=callbacks, verbose=1
    )

    # Evaluate
    y_pred = np.argmax(model.predict(X_test), axis=1)

    cm = confusion_matrix(y_test, y_pred, labels=range(6))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion Matrix - 2-Layer LSTM with AutoEncoder')
    plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight'); plt.show()

    print("\nPer-class accuracy:")
    for name, acc in zip(CLASS_NAMES, np.diag(cm_norm)):
        print(f"  {name}: {acc:.4f}")
    print(f"Overall accuracy: {np.trace(cm_norm) / np.sum(cm_norm):.4f}")

    model.save('lstm_model.keras')
    joblib.dump(le, 'label_encoder.pkl')

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history.history['loss'], label='Train'); ax1.plot(history.history['val_loss'], label='Val')
    ax1.set_title('Loss'); ax1.legend()
    ax2.plot(history.history['accuracy'], label='Train'); ax2.plot(history.history['val_accuracy'], label='Val')
    ax2.set_title('Accuracy'); ax2.legend()
    plt.tight_layout(); plt.savefig('lstm_training.png', dpi=150); plt.close()


# ============================================================
# Main
# ============================================================
def main():
    # Step 1: Extract all features
    print("=" * 60)
    print("STEP 1: TDS + Feature Extraction")
    print("=" * 60)
    all_features, all_labels = extract_all_data()
    print(f"\nTotal transmissions: {len(all_features)}")
    for tech in CLASS_NAMES:
        print(f"  {tech}: {np.sum(all_labels == tech)}")

    # Save for reuse
    np.save('all_features.npy', np.array(all_features, dtype=object), allow_pickle=True)
    np.save('all_labels.npy', all_labels)
    print("Saved: all_features.npy, all_labels.npy")

    # Step 2: Train AE
    encoder, scaler = train_autoencoder(all_features)

    # Step 3: Train LSTM
    train_lstm(all_features, all_labels)

    print("\n" + "=" * 60)
    print("DONE! Files saved:")
    print("  autoencoder_model.keras, encoder_model.keras")
    print("  lstm_model.keras, scaler.pkl, label_encoder.pkl")
    print("  confusion_matrix.png")
    print("=" * 60)


if __name__ == '__main__':
    main()
