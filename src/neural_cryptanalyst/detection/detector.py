import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

class SideChannelDetector:
    def __init__(self):
        self.model = self._create_detection_model()
    def _create_detection_model(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(400,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    def extract_features(self, power_measurements):
        features = []
        features.append(np.mean(power_measurements, axis=0))
        features.append(np.std(power_measurements, axis=0))
        features.append(np.max(power_measurements, axis=0) - np.min(power_measurements, axis=0))
        fft_values = np.abs(np.fft.fft(power_measurements, axis=0))
        features.append(np.mean(fft_values, axis=0))
        return np.concatenate([f.flatten() for f in features])
    def detect_attack(self, power_measurements, threshold=0.8):
        features = self.extract_features(power_measurements)
        features = features.reshape(1, -1)
        if features.shape[1] < 400:
            features = np.pad(features, ((0, 0), (0, 400 - features.shape[1])))
        elif features.shape[1] > 400:
            features = features[:, :400]
        attack_probability = self.model.predict(features)[0][0]
        if attack_probability > threshold:
            self.trigger_countermeasures()
            return True
        return False
    def trigger_countermeasures(self):
        print("ALERT: Side-channel attack detected!")
        print("Activating countermeasures: random delays, masking, etc.")
