import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CLASS_NAMES = ['DAB', 'DVB-T', 'FM', 'GSM', 'LTE', 'TETRA']
LSTM_EPOCHS = 550
LSTM_PATIENCE = 10
LSTM_LR = 0.001
DROPOUT_RATE = 0.001
MAX_TIME_SWEEPS = 50


def pad_sequences_custom(sequences, maxlen=None):
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


def build_lstm():
    inp = keras.Input(shape=(None, 16))
    x = layers.LSTM(32, return_sequences=True, activation='relu')(inp)
    x = layers.LSTM(16, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    out = layers.Dense(6, activation='softmax')(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(LSTM_LR),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    print("=" * 60)
    print("TRAINING LSTM CLASSIFIER")
    print("=" * 60)

    # Load saved data
    print("\n[1/4] Loading data...")
    all_features = np.load('all_features.npy', allow_pickle=True)
    all_labels = np.load('all_labels.npy')
    encoder = keras.models.load_model('encoder_model.keras')
    scaler = joblib.load('scaler.pkl')
    print(f"  Transmissions: {len(all_features)}")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(all_labels)

    # Encode each transmission's feature sequence
    print("\n[2/4] Encoding sequences...")
    sequences = []
    for feat_matrix in all_features:
        scaled = scaler.transform(feat_matrix)
        encoded = encoder.predict(scaled, verbose=0)
        sequences.append(encoded)

    X, lengths = pad_sequences_custom(sequences)
    print(f"  Padded shape: {X.shape}")

    # Split
    X_train, X_test, y_train, y_test, len_train, len_test = train_test_split(
        X, y, lengths, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    for tech in CLASS_NAMES:
        if tech in le.classes_:
            print(f"    {tech}: train={np.sum(y_train == le.transform([tech])[0])}, test={np.sum(y_test == le.transform([tech])[0])}")

    # Build and train LSTM
    print("\n[3/4] Training LSTM...")
    model = build_lstm()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=LSTM_PATIENCE, restore_best_weights=True)
    ]
    history = model.fit(
        X_train, y_train,
        epochs=LSTM_EPOCHS, batch_size=32,
        validation_split=0.1,
        callbacks=callbacks, verbose=1
    )

    # Evaluate
    print("\n[4/4] Evaluating...")
    y_pred = np.argmax(model.predict(X_test), axis=1)

    cm = confusion_matrix(y_test, y_pred, labels=range(6))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix - 2-Layer LSTM with AutoEncoder')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nPer-class accuracy:")
    for name, acc in zip(CLASS_NAMES, np.diag(cm_norm)):
        print(f"  {name}: {acc:.4f}")
    print(f"Overall accuracy: {np.trace(cm_norm) / np.sum(cm_norm):.4f}")

    model.save('lstm_model.keras')
    joblib.dump(le, 'label_encoder.pkl')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history.history['loss'], label='Train')
    ax1.plot(history.history['val_loss'], label='Val')
    ax1.set_title('Loss'); ax1.legend()
    ax2.plot(history.history['accuracy'], label='Train')
    ax2.plot(history.history['val_accuracy'], label='Val')
    ax2.set_title('Accuracy'); ax2.legend()
    plt.tight_layout()
    plt.savefig('lstm_training.png', dpi=150)
    plt.close()
    print("\nSaved: lstm_model.keras, confusion_matrix.png")


if __name__ == '__main__':
    main()
