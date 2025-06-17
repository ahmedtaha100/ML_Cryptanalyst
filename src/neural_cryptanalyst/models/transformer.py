import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class WarmupSchedule(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = d_model
        self.d_model_float = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model_float) * tf.math.minimum(arg1, arg2)

class SideChannelTransformer:
    def __init__(self, trace_length, d_model=256, num_heads=8, num_classes=256):
        if trace_length <= 0 or d_model <= 0 or num_heads <= 0 or num_classes <= 0:
            raise ValueError("Invalid input dimensions")
        self.trace_length = trace_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.model = self._build_model()
    def positional_encoding(self, length, depth):
        positions = np.arange(length)[:, np.newaxis].astype('float32')
        depths = (np.arange(depth)[np.newaxis, :] // 2 * 2).astype('float32')
        angle_rates = 1 / (10000**(depths / depth))
        angle_rads = positions * angle_rates
        pos_encoding = np.zeros((length, depth), dtype='float32')
        pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.constant(pos_encoding, dtype=tf.float32)
    def transformer_block(self, x, ff_dim, num_heads, dropout_rate=0.1):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=self.d_model // num_heads)(x, x)
        attn_output = Dropout(dropout_rate)(attn_output)
        out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)
        ffn = tf.keras.Sequential([Dense(ff_dim, activation='relu'), Dropout(dropout_rate), Dense(self.d_model)])
        ffn_output = ffn(out1)
        out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        return out2
    def _build_model(self):
        inputs = Input(shape=(self.trace_length, 1))
        x = Dense(self.d_model)(inputs)
        pos_encoding = self.positional_encoding(self.trace_length, self.d_model)
        x = x + pos_encoding
        x = self.transformer_block(x, ff_dim=2048, num_heads=self.num_heads)
        x = self.transformer_block(x, ff_dim=2048, num_heads=self.num_heads)
        x = self.transformer_block(x, ff_dim=2048, num_heads=self.num_heads)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax', dtype='float32')(x)
        return Model(inputs=inputs, outputs=outputs)
    def compile_model(self, learning_rate=0.0001, warmup_steps=4000):
        lr_schedule = WarmupSchedule(self.d_model, warmup_steps)
        optimizer = Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    def get_attention_weights(self, traces):
        traces = traces.astype('float32')
        attention_layers = [layer for layer in self.model.layers if isinstance(layer, MultiHeadAttention)]
        if not attention_layers:
            raise ValueError("No attention layers found in model")
        attention_model = Model(inputs=self.model.input, outputs=[layer.output for layer in attention_layers])
        return attention_model.predict(traces)

    def save_model(self, filepath: str) -> None:
        """Save model weights and architecture"""
        self.model.save(filepath)
        import json
        config = {
            'trace_length': self.trace_length,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
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

    @classmethod
    def create_gpam_variant(cls, trace_length: int, num_classes: int = 256):
        """Create GPAM-inspired variant with temporal patches."""
        return cls(trace_length=trace_length, d_model=512, num_heads=16, num_classes=num_classes)

    @classmethod
    def create_transnet_variant(cls, trace_length: int, num_classes: int = 256):
        """Create TransNet-inspired variant for shift-invariant analysis."""
        return cls(trace_length=trace_length, d_model=256, num_heads=8, num_classes=num_classes)

    def add_shift_invariance_augmentation(self, shift_range: int = 100):
        """Add shift-invariance augmentation layer (TransNet-style)."""
        from tensorflow.keras.layers import Lambda
        import tensorflow as tf

        def random_shift(x):
            if tf.keras.backend.learning_phase():
                batch_size = tf.shape(x)[0]
                shifts = tf.random.uniform((batch_size,), -shift_range, shift_range, dtype=tf.int32)

                shifted = []
                for i in range(batch_size):
                    shifted.append(tf.roll(x[i], shifts[i], axis=0))
                return tf.stack(shifted)
            return x

        shift_layer = Lambda(random_shift, name='shift_augmentation')

        inputs = self.model.input
        shifted = shift_layer(inputs)

        x = self.model.layers[1](shifted)
        for layer in self.model.layers[2:]:
            x = layer(x)

        self.model = Model(inputs=inputs, outputs=x)
