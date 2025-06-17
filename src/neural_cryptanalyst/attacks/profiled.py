import numpy as np
from typing import Optional, Union, Tuple
import tensorflow as tf

from ..preprocessing import TracePreprocessor, FeatureSelector, TraceAugmenter
from ..models import SideChannelCNN, SideChannelLSTM

class ProfiledAttack:
    """Profiled side-channel attack framework."""

    def __init__(self, model: Optional[Union[SideChannelCNN, SideChannelLSTM]] = None,
                 preprocessor: Optional[TracePreprocessor] = None,
                 feature_selector: Optional[FeatureSelector] = None,
                 augmenter: Optional[TraceAugmenter] = None):
        self.model = model
        self.preprocessor = preprocessor or TracePreprocessor()
        self.feature_selector = feature_selector or FeatureSelector()
        self.augmenter = augmenter
        self.key_predictions = None

    def prepare_data(self, traces: np.ndarray, labels: np.ndarray,
                     num_features: int = 1000, augment: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess traces, select features, and apply augmentation."""
        if not self.preprocessor._fitted:
            self.preprocessor.fit(traces)
        processed = self.preprocessor.preprocess_traces(traces)

        _, selected = self.feature_selector.select_poi_sost(processed, labels, num_poi=num_features)

        if augment and self.augmenter:
            selected, labels = self.augmenter.augment_batch(selected, labels)

        if len(selected.shape) == 2:
            selected = selected.reshape(selected.shape[0], selected.shape[1], 1)

        return selected, labels

    def train_model(self, traces: np.ndarray, labels: np.ndarray, validation_split: float = 0.2,
                    num_features: int = 1000, epochs: int = 100, batch_size: int = 64):
        """Train the underlying neural network model."""
        X, y = self.prepare_data(traces, labels, num_features=num_features, augment=True)
        y_onehot = tf.keras.utils.to_categorical(y, num_classes=256)

        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y_onehot[:split_idx], y_onehot[split_idx:]

        if self.model is None:
            self.model = SideChannelCNN(trace_length=X.shape[1])
            self.model.compile_model()

        history = self.model.train(X_train, y_train, X_val, y_val,
                                   epochs=epochs, batch_size=batch_size)
        return history

    def attack(self, traces: np.ndarray, num_attack_traces: int = 100,
               num_features: Optional[int] = None) -> np.ndarray:
        """Run the attack on traces and return key probability distributions."""
        if self.model is None:
            raise ValueError("Model must be trained before attack")

        if num_features is None:
            if hasattr(self.model, 'trace_length'):
                num_features = self.model.trace_length
            else:
                num_features = traces.shape[1]

        dummy_labels = np.zeros(len(traces))
        X, _ = self.prepare_data(
            traces[:num_attack_traces],
            dummy_labels[:num_attack_traces],
            num_features=num_features,
            augment=False
        )

        predictions = self.model.model.predict(X)
        self.key_predictions = predictions
        return predictions

    def analyze_attack_quality(self, traces: np.ndarray, predictions: np.ndarray,
                              correct_key: int) -> dict:
        """Analyze attack quality using multiple metrics including MI."""
        from ..attacks.metrics import calculate_mutual_information_analysis

        metrics = {}

        mi = calculate_mutual_information_analysis(traces, predictions, correct_key)
        metrics['mutual_information'] = mi

        correct_probs = predictions[:, correct_key]
        other_probs = np.delete(predictions, correct_key, axis=1)

        signal = np.mean(correct_probs)
        noise = np.std(other_probs.flatten())
        metrics['prediction_snr'] = signal / (noise + 1e-10)

        metrics['discrimination_ratio'] = np.mean(correct_probs) / np.mean(other_probs)

        max_probs = np.max(predictions, axis=1)
        metrics['average_confidence'] = np.mean(max_probs)
        metrics['correct_key_confidence'] = np.mean(correct_probs)

        predicted_keys = np.argmax(predictions, axis=1)
        metrics['success_rate'] = np.mean(predicted_keys == correct_key)

        return metrics

    def attack_with_analysis(self, traces: np.ndarray, correct_key: Optional[int] = None,
                            num_attack_traces: int = 100,
                            num_features: Optional[int] = None) -> Tuple[np.ndarray, Optional[dict]]:
        """Run attack and optionally analyze quality if correct key is known."""

        predictions = self.attack(traces, num_attack_traces, num_features)

        analysis = None
        if correct_key is not None:
            analysis = self.analyze_attack_quality(
                traces[:num_attack_traces],
                predictions,
                correct_key
            )

        return predictions, analysis
