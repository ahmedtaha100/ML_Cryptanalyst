# Neural Cryptanalyst: Machine Learning-Powered Side Channel Attacks

[![Python 3.8-3.10](https://img.shields.io/badge/python-3.8--3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/AhmedTaha100/ML_Cryptanalyst/actions/workflows/tests.yml/badge.svg)](https://github.com/AhmedTaha100/ML_Cryptanalyst/actions)
[![Documentation](https://img.shields.io/badge/docs-available-green.svg)](docs/)

This repository accompanies the research paper **"The Neural Cryptanalyst: Machine Learning-Powered Side Channel Attacks"** by **Ahmed Taha** (Johns Hopkins University, 2025). The work explores how convolutional neural networks (CNNs), long short-term memory networks (LSTMs), and Transformer-based models dramatically reduce the number of power traces needed to compromise cryptographic implementations.

## 📄 Survey Paper

The full survey — **"The Neural Cryptanalyst: Machine-Learning-Powered Side-Channel Attacks – A Comprehensive Survey"** — is available:

* **Direct PDF:** [The Neural Cryptanalyst - Machine Learning-Powered Side Channel Attacks - A Comprehensive Survey.pdf](https://github.com/AhmedTaha100/ML_Cryptanalyst/blob/main/The%20Neural%20Cryptanalyst-%20Machine%20Learning-Powered%20Side%20Channel%20Attacks%20-%20A%20Comprehensive%20Survey.pdf)
* **Zenodo:** <https://doi.org/10.5281/zenodo.15694329>
* **Google Scholar:** <https://scholar.google.com/citations?user=FQ1XbKcAAAAJ&hl=en>
* **arXiv:** [Coming soon]

### 📑 How to cite

```bibtex
@misc{taha2025neural,
  author       = {Ahmed Taha},
  title        = {The Neural Cryptanalyst: Machine-Learning-Powered Side-Channel Attacks — A Comprehensive Survey},
  year         = {2025},
  howpublished = {Preprint},
  url          = {https://github.com/AhmedTaha100/ML_Cryptanalyst},
  doi          = {10.5281/zenodo.15694329}
}
```

## 🔬 Key Features

- **State-of-the-art neural architectures**: CNN, LSTM, Transformer, and hybrid models optimized for side-channel analysis
- **Automated preprocessing pipeline**: Alignment, filtering, and Points of Interest (POI) selection
- **Comprehensive attack suite**: Profiled and non-profiled attacks on AES, RSA, and ECC
- **Advanced countermeasures**: Masking, hiding, and constant-time implementations
- **Reproducible research**: Complete scripts to reproduce all paper results

## Requirements

- Python 3.8-3.10 (TensorFlow 2.14+ dropped Python 3.8 support)
- 8GB+ RAM (16GB recommended for large datasets)
- GPU with 8GB+ VRAM (optional but recommended)
- ~10GB disk space for datasets
- Ubuntu 20.04+ or Windows 10/11 (macOS supported but not tested in CI)

## Quick Start

1. **Clone and install**
```bash
git clone https://github.com/AhmedTaha100/ML_Cryptanalyst.git
cd ML_Cryptanalyst
pip install -e .  # This installs all dependencies from setup.py
```
2. **Install test requirements (optional)**
```bash
pip install -r requirements-dev.txt
```

3. **Run the demonstration script**
```bash
python neural_cryptanalyst_cli.py
```

4. **Run unit tests**
```bash
chmod +x scripts/setup_test_env.sh
./scripts/setup_test_env.sh
pytest
```


## Dataset Setup

### Option 1: ASCAD Dataset (Recommended for Real Experiments)
1. Visit the [ASCAD GitHub repository](https://github.com/ANSSI-FR/ASCAD)
2. Download `ASCAD_data.zip` from their releases
3. Extract `ASCAD.h5` to `./ASCAD_data/` directory

### Option 2: Generate Synthetic Datasets (For Testing)
```bash
python -m neural_cryptanalyst.datasets.download
```

## Project Structure

```
ML_Cryptanalyst/
├── src/neural_cryptanalyst/
│   ├── models/              # CNN, LSTM, Transformer architectures
│   ├── attacks/             # Profiled and non-profiled attacks
│   ├── preprocessing/       # Trace alignment, filtering, POI selection
│   ├── countermeasures/     # Masking, hiding, constant-time implementations
│   ├── datasets/            # Dataset loaders and utilities
│   ├── detection/           # ML-based attack detection
│   └── visualization/       # Plotting utilities
├── examples/                # Basic and advanced usage examples
├── paper_reproduction/      # Scripts to reproduce all paper results
└── tests/                   # Comprehensive test suite
```

## Using the API

### Basic Attack Example

```python
from neural_cryptanalyst.attacks.profiled import ProfiledAttack
from neural_cryptanalyst.models import SideChannelCNN
from neural_cryptanalyst.datasets import ASCADDataset

# Load dataset
dataset = ASCADDataset()
training_traces, training_labels = dataset.load_ascad_v1("ASCAD_data/ASCAD.h5")

# Train and execute attack
attack = ProfiledAttack(model=SideChannelCNN(trace_length=700))
attack.train_model(training_traces[:45000], training_labels[:45000])

# Attack
test_traces, _ = dataset.get_attack_set("ASCAD_data/ASCAD.h5")
predictions = attack.attack(test_traces[:100])
```

### With Preprocessing Pipeline

```python
from neural_cryptanalyst import TracePreprocessor, FeatureSelector
from neural_cryptanalyst.datasets import ASCADDataset

# Load dataset first
dataset = ASCADDataset()
training_traces, training_labels = dataset.load_ascad_v1('ASCAD_data/ASCAD.h5')

# Preprocess traces
preprocessor = TracePreprocessor()
preprocessor.fit(training_traces)
processed = preprocessor.preprocess_traces(training_traces)

# Select points of interest
selector = FeatureSelector()
poi_indices, selected_traces = selector.select_poi_sost(processed, training_labels, num_poi=1000)

# Train model on selected features
from neural_cryptanalyst.attacks.profiled import ProfiledAttack
from neural_cryptanalyst.models import SideChannelCNN

attack = ProfiledAttack(model=SideChannelCNN(trace_length=1000))
attack.train_model(selected_traces, training_labels)
```
### Complete Attack Pipeline

```python
from neural_cryptanalyst import ProfiledAttack, SideChannelCNN, TracePreprocessor, FeatureSelector
from neural_cryptanalyst.datasets import ASCADDataset

# Load data
dataset = ASCADDataset()
traces, labels = dataset.load_ascad_v1("ASCAD_data/ASCAD.h5")

# Preprocess
preprocessor = TracePreprocessor()
preprocessor.fit(traces[:1000])  # Fit on subset
processed = preprocessor.preprocess_traces(traces)

# Select POIs
selector = FeatureSelector()
poi_indices, selected = selector.select_poi_sost(processed[:45000], labels[:45000], num_poi=700)

# Train attack
attack = ProfiledAttack(model=SideChannelCNN(trace_length=700))
attack.train_model(selected[:45000], labels[:45000], epochs=50)

# Execute attack
test_selected = selector.transform(processed[45000:])
predictions = attack.attack(test_selected[:100])
```

See [`examples/`](examples/) for complete end-to-end workflows and [`paper_reproduction/`](paper_reproduction/) to reproduce paper results.

## Performance

### Attack Performance (from paper)

| Attack Type | Traditional Methods | ML-Enhanced | Trace Reduction |
|-------------|---------------------|-------------|-----------------|
| Unmasked AES | 5,000-10,000 traces | 50-200 traces | 80-90% |
| First-order Masked AES | 5,000-10,000 traces | 500-1,000 traces | 80-90% |
| Second-order Masked AES | 50,000+ traces | 3,000-5,000 traces | 90%+ |

### Model Comparison

| Model | Traces Required | Success Rate¹ | Best For |
|-------|-----------------|--------------|----------|
| CNN | 500-1,000 | 70-85% | General purpose, masked implementations |
| LSTM | 700-1,200 | 60-75% | Misaligned/desynchronized traces |
| Transformer | 100-300 | 20-40% better² | Best accuracy, computational cost higher |

¹ On first-order masked AES (ASCAD dataset)  
² Improvement over CNN baseline

## Documentation

- [API Reference](docs/API_REFERENCE.md) - Detailed function and class documentation
- [Example Notebooks](notebooks/) - Interactive tutorials
- [Paper Reproduction Guide](paper_reproduction/README.md) - Reproduce all paper results

## Troubleshooting

**GPU not detected**  
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Or disable GPU
export CUDA_VISIBLE_DEVICES=-1
```

**Out of memory errors**
- Reduce batch size: `train_model(..., batch_size=32)`
- For large datasets: `TracePipeline.process_large_dataset(filepath, batch_size=1000)`
- Enable GPU memory growth:
  ```python
  import tensorflow as tf
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
      tf.config.experimental.set_memory_growth(gpus[0], True)
  ```

**Import errors**  
Install in development mode: `pip install -e .`

**Dataset download issues**
- ASCAD requires manual download from their GitHub
- DPA Contest v4 requires registration at http://www.dpacontest.org/
- Use synthetic datasets for testing: `python -m neural_cryptanalyst.datasets.download`

**TensorFlow version conflicts**
```bash
# Force compatible version
pip install 'tensorflow>=2.8.0,<2.14.0'
```

## Academic Integrity and Plagiarism Reports

In the main directory, you'll find comprehensive academic integrity verification reports:

- **[iThenticate Plagiarism Report.html](iThenticate%20Plagiarism%20Report.html)** - Professional plagiarism analysis showing 10% similarity index
- **[iThenticate Plagiarism Report.pdf](iThenticate%20Plagiarism%20Report.pdf)** - PDF version of the plagiarism analysis
- **[Grammarly Report.pdf](Grammarly%20Report.pdf)** - Additional plagiarism verification  
- **[GPTZero AI Scan - .pdf](GPTZero%20AI%20Scan%20-%20.pdf)** - AI content detection and plagiarism analysis

These reports demonstrate the originality and academic integrity of this research work.

<small>© 2025 Ahmed Taha. All rights reserved.</small>
