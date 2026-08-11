#!/usr/bin/env python3
"""
Step 3: Load Author's Models
Load pre-trained scaler, autoencoder, and LSTM classifier.
"""

import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Set paths
BASE_PATH = 'PSD-technology-classification-framework/TCpackage/resources'
SCALER_PATH = os.path.join(BASE_PATH, 'scaler/_AE16_LSTM_Scaler_.save')
ENCODER_PATH = os.path.join(BASE_PATH, 'save-DL-models/Autoencoder_DNN/TrainAllSensorHop_16_feat_mse_relu/saved-model-49-0.0002.hdf5')
LSTM_PATH = os.path.join(BASE_PATH, 'save-DL-models/LSTM_TrainWithAE/TrainAllSensorHop_DNNAE16_LSTM_mse_relu/saved-model-110-0.97.hdf5')

def load_scaler():
    """Load the pre-trained StandardScaler."""
    print("Loading scaler...")
    scaler = joblib.load(SCALER_PATH)
    print(f"  Scaler loaded: {scaler.n_features_in_} features")
    return scaler

def build_encoder():
    """Build and load the autoencoder encoder.
    
    The weight file contains the full autoencoder (encoder + decoder).
    We build both, load weights, then extract just the encoder.
    """
    print("Building encoder...")
    
    # Build encoder
    encoder = Sequential()
    encoder.add(Dense(units=64, activation='relu', input_shape=[33]))
    encoder.add(Dense(units=32, activation='relu'))
    encoder.add(Dense(units=16, activation='relu'))
    
    # Build decoder (needed to load full autoencoder weights)
    decoder = Sequential()
    decoder.add(Dense(units=16, activation='relu', input_shape=[16]))
    decoder.add(Dense(units=32, activation='relu'))
    decoder.add(Dense(units=64, activation='relu'))
    decoder.add(Dense(units=33))
    
    # Build full autoencoder
    autoencoder = Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Build by running dummy input
    dummy_input = np.zeros((1, 33), dtype=np.float32)
    _ = autoencoder(dummy_input)
    
    print("Loading autoencoder weights...")
    autoencoder.load_weights(ENCODER_PATH)
    
    # Extract encoder from autoencoder
    encoder = autoencoder.layers[0]
    print(f"  Encoder architecture: {encoder.output_shape}")
    print("  Encoder weights loaded")
    
    return encoder

def build_lstm():
    """Build and load the LSTM classifier."""
    print("Building LSTM classifier...")
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(16, 1), return_sequences=True))
    model.add(LSTM(16, activation='relu'))
    model.add(Dense(16, activation='softmax'))
    model.add(Dropout(0.001))
    model.add(Dense(6, activation='softmax'))
    
    print(f"  LSTM architecture: {model.output_shape}")
    
    # Build the model by running a dummy input
    dummy_input = np.zeros((1, 16, 1), dtype=np.float32)
    _ = model(dummy_input)
    
    print("Loading LSTM weights...")
    model.load_weights(LSTM_PATH)
    print("  LSTM weights loaded")
    
    return model

def test_inference(scaler, encoder, model):
    """Test the inference pipeline with dummy data."""
    print("\nTesting inference pipeline...")
    
    # Create dummy 33-feature vector
    dummy_features = np.random.randn(1, 33).astype(np.float32)
    
    # Scale
    scaled = scaler.transform(dummy_features)
    print(f"  Scaled features shape: {scaled.shape}")
    
    # Encode
    encoded = encoder.predict(scaled, verbose=0)
    print(f"  Encoded features shape: {encoded.shape}")
    
    # Reshape for LSTM
    reshaped = np.reshape(encoded, (-1, 16, 1))
    print(f"  Reshaped for LSTM: {reshaped.shape}")
    
    # Predict
    predictions = model.predict(reshaped, verbose=0)
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Predicted class: {np.argmax(predictions)}")
    print(f"  Class probabilities: {predictions[0]}")
    
    return True

def main():
    print("=" * 60)
    print("Step 3: Load Author's Models")
    print("=" * 60)
    
    # Load scaler
    scaler = load_scaler()
    
    # Build and load encoder
    encoder = build_encoder()
    
    # Build and load LSTM
    lstm = build_lstm()
    
    # Test inference
    test_inference(scaler, encoder, lstm)
    
    print("\n" + "=" * 60)
    print("Models loaded successfully!")
    print("=" * 60)
    
    return scaler, encoder, lstm

if __name__ == "__main__":
    scaler, encoder, lstm = main()
