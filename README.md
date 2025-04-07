# Temporal Graph Neural Networks for Misinformation Detection

This repository contains the implementation of a research project investigating the effectiveness of Temporal Graph Neural Networks (TGNNs) for early misinformation detection on social media.

## Research Question

> How can temporal graph neural networks (TGNNs) effectively detect misinformation on social media earlier than static graph models?

## Overview

This project implements and compares three different types of graph neural networks for misinformation detection:

1. **Temporal Graph Network (TGN)**: A dynamic graph model with memory that captures the evolution of user-post interactions over time.
2. **Temporal Graph Attention Network (TGAT)**: A continuous-time attention-based model that leverages temporal information in graph representation learning.
3. **Static Graph Convolutional Network (GCN)**: A baseline model that uses discrete time snapshots without explicitly modeling temporal dynamics.

The models are evaluated on multiple metrics, including standard classification metrics (accuracy, precision, recall, F1, AUC) and early detection metrics (time-to-detection, early precision/recall/F1).

## Project Structure

```
project/
├── configs/                  # Configuration files
│   └── config.yaml           # Hyperparameters and settings
│
├── data/                     # Data storage
│   ├── raw/                  # Raw dataset files
│   │   ├── coaid/            # CoAID dataset
│   │   ├── fakenewsnet/      # FakeNewsNet dataset
│   │   └── tgb/              # Temporal Graph Benchmark
│   │
│   └── processed/            # Processed graph data
│
├── models/                   # Model implementations
│   ├── tgn.py               # Temporal Graph Network
│   ├── tgat.py              # Temporal Graph Attention Network
│   └── gcn.py               # Static Graph Convolutional Network
│
├── preprocessing/            # Data preprocessing
│   └── preprocess.py         # Scripts to process data into temporal graphs
│
├── train/                    # Training pipelines
│   └── train.py              # Training script
│
├── evaluation/               # Evaluation scripts
│   └── evaluate.py           # Metrics calculation and visualization
│
├── README.md                 # Project documentation
└── requirements.txt          # Dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tgnn-misinformation-detection.git
   cd tgnn-misinformation-detection
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   
   # Install dependencies
   yarn add numpy pandas torch torch-geometric dgl scikit-learn matplotlib seaborn networkx tqdm pyyaml nltk transformers scipy tensorboard optuna
   ```

## Usage

### 1. Data Preprocessing

Process the raw datasets into temporal graph format:

```bash
python preprocessing/preprocess.py
```

This will:
- Load data from the `data/raw` directory
- Construct temporal graphs with users and posts as nodes, and interactions as edges
- Extract features (text embeddings, user metadata, temporal information)
- Save processed data to `data/processed`

### 2. Training

Train the models on the processed data:

```bash
# Train all models on all datasets
python train/train.py

# Train specific models on specific datasets
python train/train.py --models tgn tgat --datasets coaid fakenewsnet
```

Training parameters can be configured in `configs/config.yaml`.

### 3. Evaluation

Evaluate the trained models and generate visualization plots:

```bash
# Evaluate all models on all datasets
python evaluation/evaluate.py

# Evaluate specific models on specific datasets
python evaluation/evaluate.py --models tgn gcn --datasets tgb
```

This will:
- Calculate standard metrics (accuracy, precision, recall, F1, AUC)
- Calculate early detection metrics (time-to-detection, early precision/recall)
- Generate comparison plots in the `checkpoints/plots` directory
- Save metrics to JSON files for further analysis

## Models

### Temporal Graph Network (TGN)

The TGN model combines memory modules with graph neural networks to capture the dynamic evolution of the graph. It consists of:

- Memory module for tracking node states over time
- Temporal encoding for timestamps
- Graph attention layers for node embedding
- Classification layer for misinformation detection

### Temporal Graph Attention Network (TGAT)

TGAT uses a continuous-time attention mechanism to model temporal dependencies in the graph:

- Time encoding that transforms timestamps into high-dimensional embeddings
- Multi-head attention mechanism that incorporates temporal information
- Layer normalization and feed-forward networks
- Classification layer for misinformation prediction

### Static Graph Convolutional Network (GCN)

The GCN baseline aggregates information over discrete snapshots of the graph:

- Multiple GCN layers with skip connections
- Snapshot aggregation for temporal data (mean, max, or last)
- Classification layer for misinformation detection

## Datasets

The project uses the following datasets:

1. **CoAID**: COVID-19 misinformation dataset with news articles, social media posts, and user engagement
2. **FakeNewsNet**: Comprehensive dataset with news content, social context, and spatiotemporal information
3. **TGB**: Temporal Graph Benchmark with dynamic graph structures

## Results

The evaluation results demonstrate that temporal graph models (TGN and TGAT) outperform static GCN in early misinformation detection:

- Lower time-to-detection (detecting misinformation earlier)
- Higher precision and recall in early time windows
- Better overall classification performance (F1 score and AUC)

Detailed results and visualizations can be found in the `checkpoints/plots` directory after running the evaluation script.

## Configuration

The `configs/config.yaml` file contains all hyperparameters and settings, including:

- Model architectures and dimensions
- Training parameters (learning rate, batch size, etc.)
- Dataset and preprocessing settings
- Evaluation metrics configuration

## References

- Rossi, E., Chamberlain, B., et al. (2020). Temporal Graph Networks for Deep Learning on Dynamic Graphs. [arXiv:2006.10637](https://arxiv.org/abs/2006.10637)
- Xu, D., Ruan, C., et al. (2020). Inductive Representation Learning on Temporal Graphs. [arXiv:2002.07962](https://arxiv.org/abs/2002.07962)
- Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR 2017.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue in the GitHub repository.
