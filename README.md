# Misinformation Detection using Temporal Graph Networks

This repository contains code for detecting misinformation using temporal graph networks. The project compares the performance of temporal models (TGN) with static graph models (GCN) on misinformation detection tasks.

## Setup

1. Clone this repository
2. Install dependencies:

```
pip install torch numpy pandas matplotlib scikit-learn nltk tqdm pyyaml
```

3. Download NLTK resources:

```
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Data

The code works with two datasets:
- CoAID dataset
- FakeNewsNet dataset

These datasets are not included in the repository. Download them separately and place in the following directory structure:

```
data/
  raw/
    coaid/
      [coaid files]
    fakenewsnet/
      [fakenewsnet files]
  processed/
    [processed files will be saved here]
```

## Running Experiments

### Preprocessing

To preprocess the full datasets:

```
python preprocess_full_datasets.py
```

### Quick Experiment

For a quick demonstration comparing temporal vs. static models on a small subset of data:

```
python quick_experiment.py
```

### Full Pipeline

To run the complete pipeline with both models:

```
python run_full_pipeline.py
```

## Results

Results will be saved in the `results/` directory, including performance metrics and visualizations.

## Configuration

Model and training parameters can be modified in `configs/config.yaml`.
