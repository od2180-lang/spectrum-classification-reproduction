import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

from tds import TransmissionDetectionSystem
from feature_extraction import FeatureExtractor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATASET_PATH = 'dataset/opt/shared/alessio/Data_Transmitters_Identification/spectrum_bands_2'
CHUNK_SIZE = 215
AE_EPOCHS = 550
AE_PATIENCE = 10
AE_LR = 0.001


def extract_label_from_filename(filename):
    for tech in ['dab', 'dvbt', 'fm', 'gsm', 'lte', 'tetra']:
        if tech in filename.lower():
            return {'dab': 'DAB', 'dvbt': 'DVB-T', 'fm': 'FM',
                    'gsm': 'GSM', 'lte': 'LTE', 'tetra': 'TETRA'}[tech]
    return None


def extract_all_features(dataset_path):
    """Extract 32-feature vectors from all time sweeps of all transmissions."""
    tds = TransmissionDetectionSystem()
    extractor = FeatureExtractor()
    all_features = []

    count = 0
    for root, dirs, files in os.walk(dataset_path):
        for f in sorted(files):
            if not f.endswith('.npy'):
                continue
            if extract_label_from_filename(f) is None:
                continue

            file_path = os.path.join(root, f)
            data = np.load(file_path, allow_pickle=True)
            transmissions = tds.detect_transmissions(data)

            for start, end in transmissions:
                tx_data = data[:, start:end + 1]
                tx_width = tx_data.shape[1]

                if tx_width < CHUNK_SIZE:
                    padded = np.zeros((tx_data.shape[0], CHUNK_SIZE))
                    padded[:, :tx_width] = tx_data
                    tx_data = padded
                else:
                    tx_data = tx_data[:, :CHUNK_SIZE]

                for k in range(tx_data.shape[0]):
                    features = extractor.extract_features(tx_data[k, :])
                    if np.all(np.isfinite(features)):
                        all_features.append(features)

            count += 1
            if count % 50 == 0:
                print(f"  Processed {count} files, {len(all_features)} feature vectors so far")

    return np.array(all_features)


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


def main():
    print("=" * 60)
    print("STEP 1: Training AutoEncoder (32 -> 16)")
    print("=" * 60)

    # Extract features
    print("\n[1/3] Extracting features from all transmissions...")
    all_features = extract_all_features(DATASET_PATH)
    print(f"  Total feature vectors: {all_features.shape}")
    print(f"  Feature dimension: {all_features.shape[1]}")

    # Scale
    print("\n[2/3] Scaling features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)
    features_scaled = shuffle(features_scaled, random_state=42)

    # Build and train AE
    print("\n[3/3] Training AutoEncoder...")
    autoencoder, encoder = build_autoencoder(input_dim=32, compressed_dim=16)
    autoencoder.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=AE_PATIENCE, restore_best_weights=True
        )
    ]

    history = autoencoder.fit(
        features_scaled, features_scaled,
        epochs=AE_EPOCHS,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    # Save
    autoencoder.save('autoencoder_model.keras')
    encoder.save('encoder_model.keras')
    joblib.dump(scaler, 'scaler.pkl')

    # Plot training loss
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('AutoEncoder Training')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ae_training_loss.png', dpi=150)
    plt.show()

    print(f"\nFinal train loss: {history.history['loss'][-1]:.6f}")
    print(f"Final val loss: {history.history['val_loss'][-1]:.6f}")
    print("Models saved: autoencoder_model.keras, encoder_model.keras, scaler.pkl")


if __name__ == '__main__':
    main()
