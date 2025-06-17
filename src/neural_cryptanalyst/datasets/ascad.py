import h5py
import numpy as np

class ASCADDataset:
    def __init__(self):
        self.traces = None
        self.labels = None
        self.metadata = None

    def load_ascad_v1(self, filepath: str, fixed_key: bool = True):
        """Load ASCAD v1 database"""
        with h5py.File(filepath, 'r') as f:
            self.traces = np.array(f['Profiling_traces/traces'])
            self.metadata = np.array(f['Profiling_traces/metadata'])
            self.labels = self.metadata['key'][2] if fixed_key else self.metadata['key']
        return self.traces, self.labels

    def get_attack_set(self, filepath: str):
        """Load attack traces"""
        with h5py.File(filepath, 'r') as f:
            attack_traces = np.array(f['Attack_traces/traces'])
            attack_metadata = np.array(f['Attack_traces/metadata'])
        return attack_traces, attack_metadata
