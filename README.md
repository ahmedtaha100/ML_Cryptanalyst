# Neural Cryptanalyst: Machine Learning-Powered Side Channel Attacks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/ahmedtaha100/neural_cryptanalyst/actions/workflows/tests.yml/badge.svg)](https://github.com/ahmedtaha100/neural_cryptanalyst/actions)
[![Documentation](https://img.shields.io/badge/docs-available-green.svg)](docs/)

This repository accompanies the research paper **"The Neural Cryptanalyst: Machine Learning-Powered Side Channel Attacks"** by **Ahmed Taha** (Johns Hopkins University, May 4, 2025). The work explores how convolutional neural networks (CNNs), long short-term memory networks (LSTMs), and Transformer-based models dramatically reduce the number of power traces needed to compromise cryptographic implementations.

The paper can be found [here](https://scholar.google.com/citations?user=FQ1XbKcAAAAJ&hl=en) on my Google Scholar.

## Requirements

- Python 3.8-3.10 (TensorFlow 2.8+ compatibility)
- 8GB+ RAM (16GB recommended for large datasets)
- GPU with 8GB+ VRAM (optional but recommended)
- ~5GB disk space for datasets

## Quick start

1. **Clone and install dependencies**
   ```bash
   git clone https://github.com/ahmedtaha100/neural_cryptanalyst.git
   cd neural_cryptanalyst
   pip install -r requirements.txt
   ```

2. **(Optional) Install test requirements**
   ```bash
   pip install -r requirements-dev.txt
   ```

3. **Run the demonstration script**
   ```bash
   python neural_cryptanalyst_cli.py
   ```

4. **Run unit tests**
   ```bash
   ./scripts/setup_test_env.sh
   pytest
   ```

## Project Structure

```
neural_cryptanalyst/
├── src/neural_cryptanalyst/
│   ├── models/              # CNN, LSTM, Transformer architectures
│   ├── attacks/             # Profiled and non-profiled attacks
│   ├── preprocessing/       # Trace alignment, filtering, POI selection
│   └── countermeasures/     # Masking, hiding, constant-time implementations
├── examples/                # Basic and advanced usage examples
├── paper_reproduction/      # Scripts to reproduce all paper results
└── tests/                   # Comprehensive test suite
```

## Using the API

### Basic Attack Example

```python
from neural_cryptanalyst.attacks.profiled import ProfiledAttack
from neural_cryptanalyst.models import SideChannelCNN

# Train and execute attack
attack = ProfiledAttack(model=SideChannelCNN(trace_length=5000))
attack.train_model(training_traces, training_labels)
predictions = attack.attack(test_traces)
```

### With Preprocessing Pipeline

```python
from neural_cryptanalyst import TracePreprocessor, FeatureSelector

# Preprocess traces
preprocessor = TracePreprocessor()
preprocessor.fit(training_traces)
processed = preprocessor.preprocess_traces(training_traces)

# Select points of interest
selector = FeatureSelector()
poi_indices, selected_traces = selector.select_poi_sost(processed, training_labels, num_poi=1000)

# Train model on selected features
attack.train_model(selected_traces, training_labels)
```

See [`examples/`](examples/) for complete end-to-end workflows and [`paper_reproduction/`](paper_reproduction/) to reproduce paper results.

## Performance

| Model | Traces Required | Success Rate | Best For |
|-------|-----------------|--------------|----------|
| CNN | 500-1,000 | 92% | General purpose, masked implementations |
| LSTM | 700-1,200 | 90% | Misaligned/desynchronized traces |
| Transformer | 100-300 | 96% | Best accuracy, computational cost higher |

*Results on ASCAD database with first-order masked AES implementation*

## Documentation

- [API Reference](docs/API_REFERENCE.md) - Detailed function and class documentation
- [Example Notebooks](notebooks/) - Interactive tutorials
- [Paper Reproduction Guide](paper_reproduction/README.md) - Reproduce all paper results

## Troubleshooting

**GPU not detected**  
```bash
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
# Or disable GPU: export CUDA_VISIBLE_DEVICES=-1
```

**Out of memory errors**
- Reduce batch size: `train_model(..., batch_size=32)`
- For large datasets: `TracePipeline.process_large_dataset(filepath, batch_size=1000)`

**Import errors**  
Install in development mode: `pip install -e .`

**Dataset download issues**
```bash
# Download ASCAD dataset
python -m neural_cryptanalyst.datasets.download
# DPA Contest v4 requires manual download from http://www.dpacontest.org/
```

## Citation

If you use this code, please cite:

```
@article{taha2025neural,
  title={Neural Cryptanalyst: Machine Learning-Powered Side Channel Attacks},
  author={Taha, Ahmed},
  year={2025}
}
```

<small>© 2025 Ahmed Taha. All rights reserved.</small>
