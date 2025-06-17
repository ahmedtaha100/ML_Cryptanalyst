import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, AveragePooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class SideChannelCNNLSTM:
    def __init__(self, trace_length, num_classes=256):
        if trace_length <= 0 or num_classes <= 0:
            raise ValueError("Invalid input dimensions")
        self.trace_length = trace_length
        self.num_classes = num_classes
        self.model = self._build_model()
    def _build_model(self):
        model = Sequential([
            Conv1D(64, 11, activation='relu', padding='same', input_shape=(self.trace_length, 1), dtype='float32'),
            BatchNormalization(),
            AveragePooling1D(2),
            Dropout(0.2),
            Conv1D(128, 11, activation='relu', padding='same'),
            BatchNormalization(),
            AveragePooling1D(2),
            Dropout(0.2),
            Conv1D(256, 11, activation='relu', padding='same'),
            BatchNormalization(),
            LSTM(256, return_sequences=True),
            Dropout(0.25),
            BatchNormalization(),
            LSTM(512, return_sequences=False),
            Dropout(0.25),
            BatchNormalization(),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax', dtype='float32')
        ])
        return model
    def compile_model(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def save_model(self, filepath: str) -> None:
        """Save model weights and architecture"""
        self.model.save(filepath)
        import json
        config = {
            'trace_length': self.trace_length,
            'num_classes': self.num_classes
        }
        with open(filepath + '_config.json', 'w') as f:
            json.dump(config, f)

    @classmethod
    def load_model(cls, filepath: str):
        """Load model from saved files"""
        import tensorflow as tf
        import json
        with open(filepath + '_config.json', 'r') as f:
            config = json.load(f)
        instance = cls(**config)
        instance.model = tf.keras.models.load_model(filepath)
        return instance
