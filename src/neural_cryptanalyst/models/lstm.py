import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class SideChannelLSTM:
    """Recurrent neural network model for side-channel analysis."""

    def __init__(self, trace_length: int, num_features: int = 1,
                 num_classes: int = 256, bidirectional: bool = False):
        self.trace_length = trace_length
        self.num_features = num_features
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.model = self._build_bidirectional_model() if bidirectional else self._build_model()

    def _build_model(self) -> Sequential:
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.trace_length, self.num_features)),
            Dropout(0.25),
            BatchNormalization(),

            LSTM(256, return_sequences=True),
            Dropout(0.25),
            BatchNormalization(),

            LSTM(512, return_sequences=False),
            Dropout(0.25),
            BatchNormalization(),

            Dense(1024, activation='relu'),
            Dropout(0.5),
            BatchNormalization(),

            Dense(self.num_classes, activation='softmax')
        ])
        return model

    def _build_bidirectional_model(self) -> Sequential:
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=(self.trace_length, self.num_features)),
            Dropout(0.25),
            BatchNormalization(),

            Bidirectional(LSTM(256, return_sequences=True)),
            Dropout(0.25),
            BatchNormalization(),

            Bidirectional(LSTM(512, return_sequences=False)),
            Dropout(0.25),
            BatchNormalization(),

            Dense(1024, activation='relu'),
            Dropout(0.5),
            BatchNormalization(),

            Dense(self.num_classes, activation='softmax')
        ])
        return model

    def compile_model(self, learning_rate: float = 0.0001, clipnorm: float = 1.0):
        optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val, epochs: int = 100, batch_size: int = 64):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def save_model(self, filepath: str) -> None:
        """Save model weights and architecture"""
        self.model.save(filepath)
        import json
        config = {
            'trace_length': self.trace_length,
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'bidirectional': self.bidirectional
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
