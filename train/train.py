"""
Training module for misinformation detection models.

This module implements the training pipeline for temporal graph neural networks
and baseline models for misinformation detection.
"""

import os
import sys
import yaml
import pickle
import numpy as np
import time
import json
from datetime import datetime
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn import BCELoss, BCEWithLogitsLoss
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tgn import TemporalGraphNetwork
from models.tgat import TemporalGraphAttentionNetwork
from models.gcn import GraphConvolutionalNetwork


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Based on https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, gamma=2.0, alpha=0.25):
        """
        Initialize focal loss.
        
        Args:
            gamma (float): Focusing parameter. Higher values give more weight to hard examples.
            alpha (float): Weighting factor for the positive class.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, pred, target):
        """
        Compute focal loss.
        
        Args:
            pred (torch.Tensor): Predicted probabilities.
            target (torch.Tensor): Target labels (0 or 1).
            
        Returns:
            torch.Tensor: Computed loss.
        """
        # Binary cross entropy
        bce = -(target * torch.log(pred + 1e-10) + (1 - target) * torch.log(1 - pred + 1e-10))
        
        # Apply focusing and weighting
        pt = pred * target + (1 - pred) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - pt).pow(self.gamma)
        
        return (focal_weight * bce).mean()


class EarlyStopping:
    """Early stopping for preventing overfitting."""
    
    def __init__(self, patience=10, verbose=False, delta=0, checkpoint_path='checkpoint.pt'):
        """
        Initialize early stopping.
        
        Args:
            patience (int): Number of epochs with no improvement to wait before stopping.
            verbose (bool): Whether to print messages.
            delta (float): Minimum change to qualify as improvement.
            checkpoint_path (str): Path to save the best model.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        """
        Check if training should be stopped.
        
        Args:
            val_loss (float): Validation loss.
            model (nn.Module): Model to save.
            
        Returns:
            bool: True if training should be stopped.
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
        return self.early_stop
    
    def save_checkpoint(self, val_loss, model):
        """
        Save model checkpoint.
        
        Args:
            val_loss (float): Validation loss.
            model (nn.Module): Model to save.
        """
        if self.verbose:
            print(f'Validation loss decreased ({val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.checkpoint_path)


class Trainer:
    """Training pipeline for misinformation detection models."""
    
    def __init__(self, config_path='configs/config.yaml'):
        """
        Initialize the trainer.
        
        Args:
            config_path (str): Path to the configuration file.
        """
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Set device
        self.device = torch.device(
            self.config['general']['device'] if torch.cuda.is_available() 
            else 'cpu'
        )
        
        # Set random seed for reproducibility
        self.seed = self.config['general']['seed']
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            
        # Initialize directories
        os.makedirs(self.config['general']['save_dir'], exist_ok=True)
        os.makedirs(self.config['general']['tensorboard_log_dir'], exist_ok=True)
        
    def load_data(self, dataset_name):
        """
        Load preprocessed dataset.
        
        Args:
            dataset_name (str): Name of the dataset to load.
            
        Returns:
            dict: Loaded graph data.
        """
        data_path = os.path.join('data', 'processed', f"{dataset_name}_processed.pkl")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Processed data not found at {data_path}. Run preprocessing first.")
        
        with open(data_path, 'rb') as f:
            graph_data = pickle.load(f)
        
        print(f"Loaded preprocessed data for {dataset_name}")
        return graph_data
    
    def split_data_temporal(self, graph_data):
        """
        Split data into train, validation, and test sets based on temporal order.
        
        Args:
            graph_data (dict): Graph data dictionary.
            
        Returns:
            tuple: (train_data, val_data, test_data)
        """
        # Get temporal split ratios from config
        split_config = self.config['data']['temporal_split']
        train_ratio = split_config['train']
        val_ratio = split_config['val']
        
        # Sort edges by time
        edge_times = graph_data['edge_times'].cpu().numpy()
        time_order = np.argsort(edge_times)
        
        # Determine cutoff indices
        n_edges = len(edge_times)
        train_cutoff = int(n_edges * train_ratio)
        val_cutoff = int(n_edges * (train_ratio + val_ratio))
        
        # Split edge indices
        train_indices = time_order[:train_cutoff]
        val_indices = time_order[train_cutoff:val_cutoff]
        test_indices = time_order[val_cutoff:]
        
        # Create splits
        def create_split(data, indices):
            split_data = data.copy()
            split_data['edge_index'] = data['edge_index'][:, indices]
            split_data['edge_attr'] = data['edge_attr'][indices]
            split_data['edge_times'] = data['edge_times'][indices]
            split_data['edge_types'] = data['edge_types'][indices]
            
            # Create snapshots specific to this split
            if 'snapshots' in data:
                # Filter snapshots to only include edges in this split
                split_snapshots = []
                for snapshot in data['snapshots']:
                    # Determine which edges in the snapshot fall within this split's time window
                    snapshot_time_window = snapshot['time_window']
                    split_edge_mask = (edge_times[indices] >= snapshot_time_window[0]) & \
                                      (edge_times[indices] < snapshot_time_window[1])
                    split_snapshot_indices = indices[split_edge_mask]
                    
                    # Create a new snapshot with just these edges
                    split_snapshot = {
                        'edge_index': data['edge_index'][:, split_snapshot_indices],
                        'edge_attr': data['edge_attr'][split_snapshot_indices],
                        'edge_types': data['edge_types'][split_snapshot_indices],
                        'time_window': snapshot_time_window
                    }
                    split_snapshots.append(split_snapshot)
                
                split_data['snapshots'] = split_snapshots
            
            return split_data
        
        train_data = create_split(graph_data, train_indices)
        val_data = create_split(graph_data, val_indices)
        test_data = create_split(graph_data, test_indices)
        
        print(f"Created temporal splits: {len(train_indices)} train edges, "
              f"{len(val_indices)} validation edges, {len(test_indices)} test edges.")
        
        return train_data, val_data, test_data
    
    def initialize_model(self, model_name, graph_data):
        """
        Initialize a model by name.
        
        Args:
            model_name (str): Name of the model to initialize.
            graph_data (dict): Graph data for model initialization.
            
        Returns:
            nn.Module: Initialized model.
        """
        model_config = self.config['models'][model_name]
        
        if model_name == 'tgn':
            model = TemporalGraphNetwork(model_config, graph_data['num_nodes'])
        elif model_name == 'tgat':
            model = TemporalGraphAttentionNetwork(model_config)
        elif model_name == 'gcn':
            model = GraphConvolutionalNetwork(model_config)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model.to(self.device)
    
    def initialize_optimizer(self, model):
        """
        Initialize optimizer and learning rate scheduler.
        
        Args:
            model (nn.Module): Model to optimize.
            
        Returns:
            tuple: (optimizer, scheduler)
        """
        train_config = self.config['training']
        
        if train_config['optimizer'] == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=train_config['learning_rate'],
                weight_decay=train_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {train_config['optimizer']}")
        
        # Initialize learning rate scheduler if enabled
        scheduler = None
        if train_config['lr_scheduler']['use']:
            scheduler_type = train_config['lr_scheduler']['type']
            
            if scheduler_type == 'reduce_on_plateau':
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=train_config['lr_scheduler']['factor'],
                    patience=train_config['lr_scheduler']['patience'],
                    verbose=True
                )
            elif scheduler_type == 'step':
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=train_config['lr_scheduler']['step_size'],
                    gamma=train_config['lr_scheduler']['factor']
                )
            elif scheduler_type == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.config['general']['num_epochs']
                )
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        return optimizer, scheduler
    
    def initialize_loss_function(self):
        """
        Initialize loss function based on configuration.
        
        Returns:
            nn.Module: Loss function.
        """
        train_config = self.config['training']
        
        if train_config['use_focal_loss']:
            return FocalLoss(gamma=train_config['focal_loss_gamma'])
        else:
            return BCELoss()
    
    def prepare_batch(self, data, device):
        """
        Prepare data for a single training step.
        
        Args:
            data (dict): Graph data.
            device (torch.device): Device to use.
            
        Returns:
            dict: Prepared data.
        """
        # Move tensors to device
        prepared_data = {
            'node_features': data['node_features'].to(device),
            'edge_index': data['edge_index'].to(device),
            'edge_attr': data['edge_attr'].to(device),
            'edge_times': data['edge_times'].to(device),
            'labels': data['labels'].to(device),
            'post_mask': data['post_mask'].to(device),
            'node_types': data['node_types'].to(device),
            'edge_types': data['edge_types'].to(device)
        }
        
        # Add snapshots if present
        if 'snapshots' in data:
            prepared_data['snapshots'] = [
                {
                    'edge_index': snapshot['edge_index'].to(device),
                    'edge_attr': snapshot['edge_attr'].to(device),
                    'edge_types': snapshot['edge_types'].to(device),
                    'time_window': snapshot['time_window']
                } for snapshot in data['snapshots']
            ]
        
        return prepared_data
    
    def train_epoch(self, model, data, optimizer, loss_fn, batch_size=32):
        """
        Train for one epoch.
        
        Args:
            model (nn.Module): Model to train.
            data (dict): Training data.
            optimizer (torch.optim.Optimizer): Optimizer.
            loss_fn (nn.Module): Loss function.
            batch_size (int): Batch size for edge processing.
            
        Returns:
            float: Average loss for this epoch.
        """
        model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # For temporal models, process edges in chronological order
        edge_times = data['edge_times'].cpu().numpy()
        time_order = np.argsort(edge_times)
        
        # Split into batches
        num_edges = len(edge_times)
        num_batches_total = (num_edges + batch_size - 1) // batch_size
        
        progress_bar = tqdm(range(0, num_edges, batch_size), desc="Training")
        
        for start_idx in progress_bar:
            # Get batch indices
            end_idx = min(start_idx + batch_size, num_edges)
            batch_indices = time_order[start_idx:end_idx]
            
            # Create batch data
            batch_data = {
                'node_features': data['node_features'],
                'edge_index': data['edge_index'][:, batch_indices],
                'edge_attr': data['edge_attr'][batch_indices],
                'edge_times': data['edge_times'][batch_indices],
                'post_mask': data['post_mask'],
                'labels': data['labels'],
                'node_types': data['node_types'],
                'edge_types': data['edge_types'][batch_indices]
            }
            
            # For GCN, add snapshots
            if hasattr(model, 'snapshot_aggregation'):
                batch_data['snapshots'] = data['snapshots']
            
            # Move batch to device
            batch_data = self.prepare_batch(batch_data, self.device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_data)
            
            # Get actual post nodes and their labels
            post_mask = batch_data['post_mask']
            post_labels = batch_data['labels'][post_mask].float()
            
            # Calculate loss
            loss = loss_fn(predictions, post_labels)
            
            # Backward pass and optimize
            loss.backward()
            
            # Apply gradient clipping if configured
            if self.config['training']['gradient_clipping'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    self.config['training']['gradient_clipping']
                )
            
            optimizer.step()
            
            # Update progress
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': total_loss / num_batches})
        
        return total_loss / num_batches
    
    def evaluate(self, model, data, loss_fn):
        """
        Evaluate model on validation or test data.
        
        Args:
            model (nn.Module): Model to evaluate.
            data (dict): Validation/test data.
            loss_fn (nn.Module): Loss function.
            
        Returns:
            dict: Evaluation metrics.
        """
        model.eval()
        
        with torch.no_grad():
            # Prepare data
            eval_data = self.prepare_batch(data, self.device)
            
            # Forward pass
            predictions = model(eval_data)
            
            # Get post nodes and their labels
            post_mask = eval_data['post_mask']
            post_labels = eval_data['labels'][post_mask].float()
            
            # Calculate loss
            loss = loss_fn(predictions, post_labels).item()
            
            # Calculate metrics
            pred_binary = (predictions >= 0.5).float()
            accuracy = (pred_binary == post_labels).float().mean().item()
            
            # Calculate true positives, false positives, etc.
            tp = ((pred_binary == 1) & (post_labels == 1)).sum().item()
            fp = ((pred_binary == 1) & (post_labels == 0)).sum().item()
            tn = ((pred_binary == 0) & (post_labels == 0)).sum().item()
            fn = ((pred_binary == 0) & (post_labels == 1)).sum().item()
            
            # Calculate precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'loss': loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
    
    def train(self, model_name, dataset_name):
        """
        Train a model on a dataset.
        
        Args:
            model_name (str): Name of the model to train.
            dataset_name (str): Name of the dataset to use.
            
        Returns:
            nn.Module: Trained model.
        """
        # Load data
        graph_data = self.load_data(dataset_name)
        
        # Split data
        train_data, val_data, test_data = self.split_data_temporal(graph_data)
        
        # Initialize model
        model = self.initialize_model(model_name, graph_data)
        
        # Initialize optimizer and scheduler
        optimizer, scheduler = self.initialize_optimizer(model)
        
        # Initialize loss function
        loss_fn = self.initialize_loss_function()
        
        # Initialize early stopping
        checkpoint_path = os.path.join(
            self.config['general']['save_dir'],
            f"{model_name}_{dataset_name}_best.pt"
        )
        early_stopping = EarlyStopping(
            patience=self.config['general']['early_stopping_patience'],
            verbose=True,
            checkpoint_path=checkpoint_path
        )
        
        # Initialize TensorBoard writer
        log_dir = os.path.join(
            self.config['general']['tensorboard_log_dir'],
            f"{model_name}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        writer = SummaryWriter(log_dir)
        
        # Training loop
        num_epochs = self.config['general']['num_epochs']
        batch_size = self.config['training']['batch_size']
        
        print(f"Starting training {model_name} on {dataset_name} for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Train for one epoch
            start_time = time.time()
            train_loss = self.train_epoch(model, train_data, optimizer, loss_fn, batch_size)
            epoch_time = time.time() - start_time
            
            # Evaluate on validation set
            val_metrics = self.evaluate(model, val_data, loss_fn)
            
            # Update learning rate scheduler if using
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            
            # Log metrics
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('Metrics/accuracy', val_metrics['accuracy'], epoch)
            writer.add_scalar('Metrics/precision', val_metrics['precision'], epoch)
            writer.add_scalar('Metrics/recall', val_metrics['recall'], epoch)
            writer.add_scalar('Metrics/f1', val_metrics['f1'], epoch)
            
            # Print progress
            if epoch % self.config['general']['log_interval'] == 0:
                print(f"Epoch {epoch}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Val F1: {val_metrics['f1']:.4f} | "
                      f"Time: {epoch_time:.2f}s")
            
            # Check early stopping
            if early_stopping(val_metrics['loss'], model):
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Load best model and evaluate on test set
        model.load_state_dict(torch.load(checkpoint_path))
        test_metrics = self.evaluate(model, test_data, loss_fn)
        
        print(f"\nTest Results for {model_name} on {dataset_name}:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        print(f"F1 Score: {test_metrics['f1']:.4f}")
        
        # Save test metrics
        metrics_path = os.path.join(
            self.config['general']['save_dir'],
            f"{model_name}_{dataset_name}_metrics.json"
        )
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)
        
        return model
    
    def run_experiment(self, models=None, datasets=None):
        """
        Run experiments with specified models and datasets.
        
        Args:
            models (list, optional): List of models to train. If None, uses all models in config.
            datasets (list, optional): List of datasets to use. If None, uses all datasets in config.
        """
        if models is None:
            models = list(self.config['models'].keys())
        
        if datasets is None:
            datasets = self.config['data']['datasets']
        
        results = {}
        
        for dataset_name in datasets:
            results[dataset_name] = {}
            
            for model_name in models:
                print(f"\n{'='*50}")
                print(f"Training {model_name} on {dataset_name}")
                print(f"{'='*50}\n")
                
                start_time = time.time()
                
                # Train model
                trained_model = self.train(model_name, dataset_name)
                
                # Record training time
                training_time = time.time() - start_time
                
                # Load test metrics
                metrics_path = os.path.join(
                    self.config['general']['save_dir'],
                    f"{model_name}_{dataset_name}_metrics.json"
                )
                with open(metrics_path, 'r') as f:
                    test_metrics = json.load(f)
                
                # Add training time to metrics
                test_metrics['training_time'] = training_time
                
                # Save updated metrics
                with open(metrics_path, 'w') as f:
                    json.dump(test_metrics, f, indent=4)
                
                results[dataset_name][model_name] = test_metrics
        
        # Save overall results
        results_path = os.path.join(
            self.config['general']['save_dir'],
            "experiment_results.json"
        )
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Print summary
        print("\nExperiment Results Summary:")
        for dataset_name in results:
            print(f"\nDataset: {dataset_name}")
            for model_name in results[dataset_name]:
                metrics = results[dataset_name][model_name]
                print(f"  {model_name}:")
                print(f"    F1 Score: {metrics['f1']:.4f}")
                print(f"    Training Time: {metrics['training_time']:.2f} seconds")


if __name__ == "__main__":
    parser = ArgumentParser(description="Train misinformation detection models")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--models", type=str, nargs="+", 
                       help="Models to train (space-separated)")
    parser.add_argument("--datasets", type=str, nargs="+",
                       help="Datasets to use (space-separated)")
    args = parser.parse_args()
    
    trainer = Trainer(config_path=args.config)
    trainer.run_experiment(models=args.models, datasets=args.datasets)
