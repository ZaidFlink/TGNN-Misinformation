#!/usr/bin/env python
"""
Pipeline script for running misinformation detection experiments.

This script provides a simple interface to run the complete pipeline:
1. Preprocess datasets
2. Train models
3. Evaluate models
"""

import os
import argparse
import yaml
import subprocess
from preprocessing.preprocess import DataPreprocessor
from train.train import Trainer
from evaluation.evaluate import Evaluator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run misinformation detection pipeline')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--preprocess', action='store_true',
                        help='Run preprocessing step')
    parser.add_argument('--train', action='store_true',
                        help='Run training step')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation step')
    parser.add_argument('--models', type=str, nargs='+', 
                        choices=['tgn', 'tgat', 'gcn', 'all'],
                        default=['all'], 
                        help='Models to train/evaluate')
    parser.add_argument('--datasets', type=str, nargs='+',
                        choices=['coaid', 'fakenewsnet', 'tgb', 'all'],
                        default=['all'],
                        help='Datasets to use')
    return parser.parse_args()

def expand_model_list(models):
    """Expand 'all' in model list to all available models."""
    if 'all' in models:
        return ['tgn', 'tgat', 'gcn']
    return models

def expand_dataset_list(datasets, config):
    """Expand 'all' in dataset list to all configured datasets."""
    if 'all' in datasets:
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg['data']['datasets']
    return datasets

def check_data_availability(datasets):
    """Check if data is available for processing."""
    available_datasets = []
    for dataset in datasets:
        base_path = os.path.join('data', 'raw', dataset)
        if not os.path.exists(base_path):
            print(f"Warning: Raw data for {dataset} not found at {base_path}")
            continue
            
        # Check for required files
        posts_path = os.path.join(base_path, 'posts.csv')
        interactions_path = os.path.join(base_path, 'interactions.csv')
        
        if os.path.exists(posts_path) and os.path.exists(interactions_path):
            available_datasets.append(dataset)
        else:
            print(f"Warning: Missing required files for {dataset}")
    
    return available_datasets

def preprocess_data(datasets, config_path):
    """Run preprocessing for selected datasets."""
    print(f"Preprocessing datasets: {', '.join(datasets)}")
    
    preprocessor = DataPreprocessor(config_path=config_path)
    
    for dataset in datasets:
        print(f"Processing {dataset}...")
        preprocessor.process_dataset(dataset)
    
    print("Preprocessing complete")

def train_models(models, datasets, config_path):
    """Train selected models on selected datasets."""
    print(f"Training models: {', '.join(models)} on datasets: {', '.join(datasets)}")
    
    trainer = Trainer(config_path=config_path)
    
    for dataset in datasets:
        print(f"Training on {dataset}...")
        for model in models:
            print(f"Training {model}...")
            trainer.train(model, dataset)
    
    print("Training complete")

def evaluate_models(models, datasets, config_path):
    """Evaluate selected models on selected datasets."""
    print(f"Evaluating models: {', '.join(models)} on datasets: {', '.join(datasets)}")
    
    evaluator = Evaluator(config_path=config_path)
    
    for dataset in datasets:
        print(f"Evaluating on {dataset}...")
        evaluator.evaluate_models(models, dataset)
    
    print("Evaluation complete")

def main():
    """Main entry point."""
    args = parse_args()
    
    # Expand 'all' options
    models = expand_model_list(args.models)
    datasets = expand_dataset_list(args.datasets, args.config)
    
    print("========== Misinformation Detection Pipeline ==========")
    print(f"Configuration: {args.config}")
    print(f"Models: {', '.join(models)}")
    print(f"Datasets: {', '.join(datasets)}")
    print("======================================================")
    
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Check data availability
    available_datasets = check_data_availability(datasets)
    
    if not available_datasets:
        print("Error: No valid datasets found. Please check your data directory.")
        print("Required structure: data/raw/{dataset}/posts.csv and data/raw/{dataset}/interactions.csv")
        return
    
    # Run selected pipeline steps
    if args.preprocess:
        try:
            preprocess_data(available_datasets, args.config)
        except Exception as e:
            print(f"Error during preprocessing: {str(e)}")
            import traceback
            traceback.print_exc()
            return
    
    # After preprocessing, check for processed data
    processed_datasets = []
    if args.train or args.evaluate:
        for dataset in available_datasets:
            processed_path = os.path.join('data', 'processed', f"{dataset}_processed.pkl")
            if os.path.exists(processed_path):
                processed_datasets.append(dataset)
            else:
                print(f"Warning: Processed data for {dataset} not found. Did you run preprocessing?")
        
        if not processed_datasets:
            print("Error: No processed datasets found. Run preprocessing first.")
            return
        
        # Train models
        if args.train:
            try:
                train_models(models, processed_datasets, args.config)
            except Exception as e:
                print(f"Error during training: {str(e)}")
                import traceback
                traceback.print_exc()
                return
        
        # Evaluate models
        if args.evaluate:
            try:
                evaluate_models(models, processed_datasets, args.config)
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")
                import traceback
                traceback.print_exc()
                return
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main() 