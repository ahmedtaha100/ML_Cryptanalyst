import numpy as np
from typing import List, Union
from ..models import SideChannelCNN, SideChannelLSTM, SideChannelTransformer


class EnsembleAttack:
    def __init__(self, models: List[Union[SideChannelCNN, SideChannelLSTM, SideChannelTransformer]]):
        self.models = models

    def train_ensemble(self, traces, labels, **kwargs):
        """Train all models in ensemble"""
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{len(self.models)}")
            model.compile_model()
            model.train(traces, labels, **kwargs)

    def predict_ensemble(self, traces, method: str = 'average'):
        """Combine predictions from all models"""
        all_predictions = []
        for model in self.models:
            preds = model.model.predict(traces)
            all_predictions.append(preds)
        all_predictions = np.array(all_predictions)

        if method == 'average':
            return np.mean(all_predictions, axis=0)
        elif method == 'vote':
            votes = np.argmax(all_predictions, axis=-1)
            return np.array([np.bincount(votes[:, i]).argmax()
                             for i in range(traces.shape[0])])
        else:
            raise ValueError("Unknown combination method")
