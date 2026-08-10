import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from tds import TransmissionDetectionSystem
from feature_extraction import FeatureExtractor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TECHNOLOGIES = ['dab', 'dvbt', 'fm', 'gsm', 'lte', 'tetra']
TECH_MAP = {
    'dab': 'DAB', 'dvbt': 'DVB-T', 'fm': 'FM',
    'gsm': 'GSM', 'lte': 'LTE', 'tetra': 'TETRA'
}
CLASS_NAMES = ['DAB', 'DVB-T', 'FM', 'GSM', 'LTE', 'TETRA']
DATASET_PATH = 'dataset/opt/shared/alessio/Data_Transmitters_Identification/spectrum_bands_2'
CHUNK_SIZE = 215

AE_EPOCHS = 550
AE_PATIENCE = 10
AE_LR = 0.001
LSTM_EPOCHS = 550
LSTM_PATIENCE = 10
LSTM_LR = 0.001
DROPOUT_RATE = 0.001
MAX_TIME_SWEEPS = 50


def extract_label_from_filename(filename):
    filename_lower = filename.lower()
    for tech in TECHNOLOGIES:
        if tech in filename_lower:
            return TECH_MAP[tech]
    return None


def get_transmissions_sequential(dataset_path):
    """
    Extract features from ALL time sweeps of each detected transmission.
    Returns list of (feature_sequence, label) pairs.
    Each feature_sequence has shape (num_sweeps, 32).
    """
    tds = TransmissionDetectionSystem()
    extractor = FeatureExtractor()

    sequences = []
    labels = []

    for root, dirs, files in os.walk(dataset_path):
        for f in sorted(files):
            if not f.endswith('.npy'):
                continue

            label = extract_label_from_filename(f)
            if label is None:
                continue

            file_path = os.path.join(root, f)
            data = np.load(file_path, allow_pickle=True)
            transmissions = tds.detect_transmissions(data)

            for start, end in transmissions:
                tx_data = data[:, start:end + 1]
                tx_width = tx_data.shape[1]

                # Pad or truncate to 215 bins
                if tx_width < CHUNK_SIZE:
                    padded = np.zeros((tx_data.shape[0], CHUNK_SIZE))
                    padded[:, :tx_width] = tx_data
                    tx_data = padded
                else:
                    tx_data = tx_data[:, :CHUNK_SIZE]

                # Limit time sweeps
                num_sweeps = min(tx_data.shape[0], MAX_TIME_SWEEPS)
                feature_seq = []
                for k in range(num_sweeps):
                    psd_row = tx_data[k, :]
                    features = extractor.extract_features(psd_row)
                    if np.all(np.isfinite(features)):
                        feature_seq.append(features)

                if len(feature_seq) >= 5:
                    sequences.append(np.array(feature_seq))
                    labels.append(label)

    return sequences, np.array(labels)


def pad_sequences_custom(sequences, maxlen=None):
    """Pad variable-length sequences to the same length."""
    if maxlen is None:
        maxlen = max(s.shape[0] for s in sequences)
    feat_dim = sequences[0].shape[1]

    padded = np.zeros((len(sequences), maxlen, feat_dim))
    lengths = []
    for i, seq in enumerate(sequences):
        length = min(seq.shape[0], maxlen)
        padded[i, :length, :] = seq[:length, :]
        lengths.append(length)

    return padded, np.array(lengths)


def build_autoencoder(input_dim=32, compressed_dim=16):
    encoder_input = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu', name='enc_dense1')(encoder_input)
    x = layers.Dense(32, activation='relu', name='enc_dense2')(x)
    encoded = layers.Dense(compressed_dim, activation='relu', name='encoded')(x)

    x = layers.Dense(32, activation='relu', name='dec_dense1')(encoded)
    x = layers.Dense(64, activation='relu', name='dec_dense2')(x)
    decoded = layers.Dense(input_dim, activation='linear', name='decoded')(x)

    autoencoder = keras.Model(encoder_input, decoded, name='AutoEncoder')
    encoder = keras.Model(encoder_input, encoded, name='Encoder')
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=AE_LR), loss='mse')
    return autoencoder, encoder


def build_lstm_classifier(input_dim=16, num_classes=6):
    model = keras.Sequential([
        layers.Input(shape=(None, input_dim)),
        layers.LSTM(32, return_sequences=True, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001), name='lstm1'),
        layers.LSTM(16, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001), name='lstm2'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu', name='dense1'),
        layers.Dropout(DROPOUT_RATE, name='dropout'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LSTM_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_confusion_matrix(y_true, y_pred, class_names):
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix - 2-Layer LSTM with AutoEncoder', fontsize=14)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

    per_class_acc = np.diag(cm_normalized)
    print(f"\nPer-class accuracy:")
    for name, acc in zip(class_names, per_class_acc):
        print(f"  {name}: {acc:.4f}")
    print(f"Overall accuracy: {np.trace(cm_normalized) / np.sum(cm_normalized):.4f}")


def main():
    print("=" * 60)
    print("WIRELESS TECHNOLOGY CLASSIFICATION")
    print("Reproducing Figure 11: 2-Layer LSTM with AutoEncoder")
    print("=" * 60)

    # Step 1: Extract sequential features
    print("\n[1/5] Running TDS and extracting sequential features...")
    sequences, y = get_transmissions_sequential(DATASET_PATH)
    print(f"  Total transmissions: {len(sequences)}")
    print(f"  Feature dimension: {sequences[0].shape[1]}")
    print(f"  Label distribution:")
    for tech in CLASS_NAMES:
        count = np.sum(y == tech)
        print(f"    {tech}: {count}")

    # Step 2: Encode labels and pad sequences
    print("\n[2/5] Encoding labels and padding sequences...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_padded, lengths = pad_sequences_custom(sequences)
    print(f"  Padded shape: {X_padded.shape}")

    X_train_seq, X_test_seq, y_train, y_test, lengths_train, lengths_test = train_test_split(
        X_padded, y_encoded, lengths, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"  Train: {len(X_train_seq)}, Test: {len(X_test_seq)}")

    # Step 3: Train AutoEncoder on individual feature vectors
    print("\n[3/5] Training AutoEncoder...")
    # Flatten all training vectors for AE training
    X_flat_train = X_train_seq.reshape(-1, X_train_seq.shape[-1])
    X_flat_test = X_test_seq.reshape(-1, X_test_seq.shape[-1])

    # Remove zero-padded rows
    mask_train = X_flat_train.sum(axis=1) != 0
    mask_test = X_flat_test.sum(axis=1) != 0
    X_flat_train = X_flat_train[mask_train]
    X_flat_test = X_flat_test[mask_test]

    scaler = StandardScaler()
    X_flat_train_scaled = scaler.fit_transform(X_flat_train)
    X_flat_test_scaled = scaler.transform(X_flat_test)

    autoencoder, encoder = build_autoencoder(input_dim=32, compressed_dim=16)
    autoencoder.summary()

    ae_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=AE_PATIENCE, restore_best_weights=True
        )
    ]
    autoencoder.fit(
        X_flat_train_scaled, X_flat_train_scaled,
        epochs=AE_EPOCHS, batch_size=32, validation_split=0.1,
        callbacks=ae_callbacks, verbose=1
    )

    # Step 4: Encode all sequences
    print("\n[4/5] Encoding sequences and training LSTM...")
    # Scale and encode each sequence
    X_train_encoded = np.zeros((X_train_seq.shape[0], X_train_seq.shape[1], 16))
    for i in range(X_train_seq.shape[0]):
        seq = X_train_seq[i, :lengths_train[i], :]
        seq_scaled = scaler.transform(seq)
        X_train_encoded[i, :lengths_train[i], :] = encoder.predict(seq_scaled, verbose=0)

    X_test_encoded = np.zeros((X_test_seq.shape[0], X_test_seq.shape[1], 16))
    for i in range(X_test_seq.shape[0]):
        seq = X_test_seq[i, :lengths_test[i], :]
        seq_scaled = scaler.transform(seq)
        X_test_encoded[i, :lengths_test[i], :] = encoder.predict(seq_scaled, verbose=0)

    print(f"  Encoded train shape: {X_train_encoded.shape}")
    print(f"  Encoded test shape: {X_test_encoded.shape}")

    # Train LSTM
    lstm_model = build_lstm_classifier(input_dim=16, num_classes=6)
    lstm_model.summary()

    lstm_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=LSTM_PATIENCE, restore_best_weights=True
        )
    ]
    lstm_model.fit(
        X_train_encoded, y_train,
        epochs=LSTM_EPOCHS, batch_size=32, validation_split=0.1,
        callbacks=lstm_callbacks, verbose=1
    )

    # Step 5: Evaluate
    print("\n[5/5] Generating confusion matrix...")
    y_pred_probs = lstm_model.predict(X_test_encoded)
    y_pred = np.argmax(y_pred_probs, axis=1)
    plot_confusion_matrix(y_test, y_pred, CLASS_NAMES)

    autoencoder.save('autoencoder_model.keras')
    encoder.save('encoder_model.keras')
    lstm_model.save('lstm_model.keras')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    print("\nModels saved.")


if __name__ == '__main__':
    main()
