#!/usr/bin/env python3
"""
Step 5: Inference Function
Classify features using pre-trained scaler, autoencoder, and LSTM.
"""

import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Technology labels
TECH_LABELS = {0: 'dab', 1: 'dvbt', 2: 'fm', 3: 'gsm', 4: 'lte', 5: 'tetra'}

def load_models():
    """Load pre-trained models."""
    base_path = 'PSD-technology-classification-framework/TCpackage/resources'
    
    # Load scaler
    scaler = joblib.load(os.path.join(base_path, 'scaler/_AE16_LSTM_Scaler_.save'))
    scaler.clip = True
    
    # Build and load encoder
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    
    encoder = Sequential()
    encoder.add(Dense(units=64, activation='relu', input_shape=[33]))
    encoder.add(Dense(units=32, activation='relu'))
    encoder.add(Dense(units=16, activation='relu'))
    
    decoder = Sequential()
    decoder.add(Dense(units=16, activation='relu', input_shape=[16]))
    decoder.add(Dense(units=32, activation='relu'))
    decoder.add(Dense(units=64, activation='relu'))
    decoder.add(Dense(units=33))
    
    autoencoder = Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    dummy = np.zeros((1, 33), dtype=np.float32)
    _ = autoencoder(dummy)
    autoencoder.load_weights(os.path.join(base_path, 'save-DL-models/Autoencoder_DNN/TrainAllSensorHop_16_feat_mse_relu/saved-model-49-0.0002.hdf5'))
    encoder = autoencoder.layers[0]
    
    # Build and load LSTM
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(16, 1), return_sequences=True))
    model.add(LSTM(16, activation='relu'))
    model.add(Dense(16, activation='softmax'))
    model.add(Dropout(0.001))
    model.add(Dense(6, activation='softmax'))
    dummy = np.zeros((1, 16, 1), dtype=np.float32)
    _ = model(dummy)
    model.load_weights(os.path.join(base_path, 'save-DL-models/LSTM_TrainWithAE/TrainAllSensorHop_DNNAE16_LSTM_mse_relu/saved-model-110-0.97.hdf5'))
    
    return scaler, encoder, model

def classify(features, scaler, encoder, model, temperature=1.0):
    """
    Classify features using pre-trained models with optional temperature scaling.
    
    Args:
        features: numpy array (num_segments, 33)
        scaler: pre-trained StandardScaler
        encoder: pre-trained autoencoder encoder
        model: pre-trained LSTM classifier
        temperature: float, temperature for scaling (1.0 = no scaling, >1.0 = softer)
    
    Returns:
        predicted_label: string (e.g., 'fm', 'tetra')
        predicted_class: int (0-5)
        predictions: numpy array (6,) - class probabilities
    """
    # Scale features
    scaled = scaler.transform(features)
    
    # Encode to 16 dimensions
    encoded = encoder.predict(scaled, verbose=0)
    
    # Reshape for LSTM: (num_segments, 16, 1)
    reshaped = np.reshape(encoded, (-1, 16, 1))
    
    # Predict
    predictions = model.predict(reshaped, verbose=0)
    
    # Average predictions across all segments
    avg_predictions = np.mean(predictions, axis=0)
    
    # Apply temperature scaling
    if temperature != 1.0:
        logits = np.log(avg_predictions + 1e-10)
        scaled_logits = logits / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        avg_predictions = exp_logits / np.sum(exp_logits)
    
    # Get predicted class
    predicted_class = np.argmax(avg_predictions)
    predicted_label = TECH_LABELS[predicted_class]
    
    return predicted_label, predicted_class, avg_predictions

def test_inference():
    """Test inference on sample data."""
    print("Testing inference...")
    
    # Load models
    print("Loading models...")
    scaler, encoder, model = load_models()
    print("Models loaded")
    
    # Load sample FM file
    from feature_extraction import extract_33_features
    
    sample_file = 'dataset/opt/shared/alessio/Data_Transmitters_Identification/spectrum_bands_2/miguel_murcia/Dec_1/SpectrumBands_85_105_fm_Esp_85_105.npy'
    data = np.load(sample_file)
    print(f"\nLoaded sample: {os.path.basename(sample_file)}")
    print(f"Data shape: {data.shape}")
    
    # Truncate to 50 time segments
    data = data[:50, :].astype(np.float32)
    
    # Extract features
    features, _ = extract_33_features(data)
    print(f"Features shape: {features.shape}")
    
    # Classify
    predicted_label, predicted_class, predictions = classify(features, scaler, encoder, model)
    
    print(f"\nPrediction: {predicted_label} (class {predicted_class})")
    print(f"Ground truth: fm")
    print(f"Match: {predicted_label == 'fm'}")
    print(f"\nClass probabilities:")
    for i, tech in TECH_LABELS.items():
        print(f"  {tech}: {predictions[i]:.4f}")
    
    return predicted_label == 'fm'

if __name__ == "__main__":
    test_inference()
