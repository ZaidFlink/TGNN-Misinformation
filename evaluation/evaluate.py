"""
Evaluation module for misinformation detection models.

This module implements metrics calculation and visualization for comparing
temporal graph neural networks with static baselines.
"""

import os
import sys
import yaml
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_curve, auc, confusion_matrix
)
import torch
from argparse import ArgumentParser

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tgn import TemporalGraphNetwork
from models.tgat import TemporalGraphAttentionNetwork
from models.gcn import GraphConvolutionalNetwork


class EarlyDetectionMetrics:
    """Metrics for early detection performance evaluation."""
    
    @staticmethod
    def time_to_detection(predictions, labels, timestamps, threshold=0.5):
        """
        Calculate time to detection for true positive misinformation.
        
        Args:
            predictions (list): Model predictions over time.
            labels (list): True labels.
            timestamps (list): Prediction timestamps.
            threshold (float): Classification threshold.
            
        Returns:
            float: Average time to detection in hours.
        """
        # Find misinformation posts (true positives)
        misinformation_indices = [i for i, label in enumerate(labels) if label == 1]
        
        if not misinformation_indices:
            return float('inf')  # No misinformation found
        
        detection_times = []
        
        for idx in misinformation_indices:
            # Get first timestamp for this post
            post_start_time = timestamps[idx][0]
            
            # Find first time post was correctly classified as misinformation
            detection_time = None
            for pred, time in zip(predictions[idx], timestamps[idx]):
                if pred >= threshold:
                    detection_time = time
                    break
            
            if detection_time is not None:
                # Calculate time difference in hours
                time_diff = (detection_time - post_start_time).total_seconds() / 3600
                detection_times.append(time_diff)
            else:
                # Penalize non-detection with a large value
                detection_times.append(float('inf'))
        
        # Return average time to detection
        finite_times = [t for t in detection_times if t != float('inf')]
        if not finite_times:
            return float('inf')
        return sum(finite_times) / len(finite_times)
    
    @staticmethod
    def early_precision(predictions, labels, timestamps, time_windows, threshold=0.5):
        """
        Calculate precision at early time windows.
        
        Args:
            predictions (list): Model predictions over time.
            labels (list): True labels.
            timestamps (list): Prediction timestamps.
            time_windows (list): Time windows in hours to evaluate.
            threshold (float): Classification threshold.
            
        Returns:
            dict: Precision values at each time window.
        """
        results = {}
        
        for window in time_windows:
            window_predictions = []
            window_labels = []
            
            for idx, label in enumerate(labels):
                post_start_time = timestamps[idx][0]
                post_end_time = post_start_time + timedelta(hours=window)
                
                # Find predictions within time window
                window_prediction = None
                for pred, time in zip(predictions[idx], timestamps[idx]):
                    if time <= post_end_time:
                        window_prediction = pred
                    else:
                        break
                
                if window_prediction is not None:
                    window_predictions.append(window_prediction >= threshold)
                    window_labels.append(label)
            
            # Calculate precision if there are predictions
            if window_predictions:
                precision = precision_score(window_labels, window_predictions, zero_division=0)
                results[window] = precision
            else:
                results[window] = 0.0
        
        return results
    
    @staticmethod
    def early_recall(predictions, labels, timestamps, time_windows, threshold=0.5):
        """
        Calculate recall at early time windows.
        
        Args:
            predictions (list): Model predictions over time.
            labels (list): True labels.
            timestamps (list): Prediction timestamps.
            time_windows (list): Time windows in hours to evaluate.
            threshold (float): Classification threshold.
            
        Returns:
            dict: Recall values at each time window.
        """
        results = {}
        
        for window in time_windows:
            window_predictions = []
            window_labels = []
            
            for idx, label in enumerate(labels):
                post_start_time = timestamps[idx][0]
                post_end_time = post_start_time + timedelta(hours=window)
                
                # Find predictions within time window
                window_prediction = None
                for pred, time in zip(predictions[idx], timestamps[idx]):
                    if time <= post_end_time:
                        window_prediction = pred
                    else:
                        break
                
                if window_prediction is not None:
                    window_predictions.append(window_prediction >= threshold)
                    window_labels.append(label)
            
            # Calculate recall if there are predictions
            if window_predictions:
                recall = recall_score(window_labels, window_predictions, zero_division=0)
                results[window] = recall
            else:
                results[window] = 0.0
        
        return results
    
    @staticmethod
    def early_f1(precision_dict, recall_dict):
        """
        Calculate F1 score at early time windows.
        
        Args:
            precision_dict (dict): Precision values from early_precision.
            recall_dict (dict): Recall values from early_recall.
            
        Returns:
            dict: F1 values at each time window.
        """
        results = {}
        
        for window in precision_dict:
            p = precision_dict[window]
            r = recall_dict[window]
            
            if p + r > 0:
                f1 = 2 * p * r / (p + r)
            else:
                f1 = 0.0
                
            results[window] = f1
        
        return results


class Evaluator:
    """Evaluation pipeline for misinformation detection models."""
    
    def __init__(self, config_path='configs/config.yaml'):
        """
        Initialize the evaluator.
        
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
        
        # Early detection time windows
        self.time_windows = self.config['evaluation']['early_detection']['time_windows']
        
        # Create output directory for plots
        self.output_dir = os.path.join(self.config['general']['save_dir'], 'plots')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_model(self, model_name, dataset_name, graph_data):
        """
        Load a trained model.
        
        Args:
            model_name (str): Name of the model.
            dataset_name (str): Name of the dataset.
            graph_data (dict): Graph data for model initialization.
            
        Returns:
            nn.Module: Loaded model.
        """
        model_config = self.config['models'][model_name]
        
        # Initialize model architecture
        if model_name == 'tgn':
            model = TemporalGraphNetwork(model_config, graph_data['num_nodes'])
        elif model_name == 'tgat':
            model = TemporalGraphAttentionNetwork(model_config)
        elif model_name == 'gcn':
            model = GraphConvolutionalNetwork(model_config)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Load trained weights
        checkpoint_path = os.path.join(
            self.config['general']['save_dir'],
            f"{model_name}_{dataset_name}_best.pt"
        )
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        return model
    
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
            raise FileNotFoundError(f"Processed data not found at {data_path}")
        
        with open(data_path, 'rb') as f:
            graph_data = pickle.load(f)
        
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
        
        return train_data, val_data, test_data
    
    def prepare_data_for_model(self, data):
        """
        Prepare data for model input.
        
        Args:
            data (dict): Graph data.
            
        Returns:
            dict: Prepared data on device.
        """
        # Move tensors to device
        prepared_data = {
            'node_features': data['node_features'].to(self.device),
            'edge_index': data['edge_index'].to(self.device),
            'edge_attr': data['edge_attr'].to(self.device),
            'edge_times': data['edge_times'].to(self.device),
            'labels': data['labels'].to(self.device),
            'post_mask': data['post_mask'].to(self.device),
            'node_types': data['node_types'].to(self.device),
            'edge_types': data['edge_types'].to(self.device)
        }
        
        # Add snapshots if present
        if 'snapshots' in data:
            prepared_data['snapshots'] = [
                {
                    'edge_index': snapshot['edge_index'].to(self.device),
                    'edge_attr': snapshot['edge_attr'].to(self.device),
                    'edge_types': snapshot['edge_types'].to(self.device),
                    'time_window': snapshot['time_window']
                } for snapshot in data['snapshots']
            ]
        
        return prepared_data
    
    def evaluate_model(self, model, test_data):
        """
        Evaluate model on test data.
        
        Args:
            model (nn.Module): Model to evaluate.
            test_data (dict): Test data.
            
        Returns:
            dict: Evaluation metrics.
        """
        model.eval()
        
        with torch.no_grad():
            # Prepare data
            eval_data = self.prepare_data_for_model(test_data)
            
            # Forward pass
            predictions = model(eval_data)
            
            # Get post nodes and their labels
            post_mask = eval_data['post_mask']
            post_labels = eval_data['labels'][post_mask].float().cpu().numpy()
            predictions = predictions.cpu().numpy()
            
            # Calculate standard metrics
            pred_binary = (predictions >= 0.5).astype(float)
            accuracy = accuracy_score(post_labels, pred_binary)
            precision = precision_score(post_labels, pred_binary, zero_division=0)
            recall = recall_score(post_labels, pred_binary, zero_division=0)
            f1 = f1_score(post_labels, pred_binary, zero_division=0)
            
            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(post_labels, predictions)
            roc_auc = auc(fpr, tpr)
            
            # Compute confusion matrix
            cm = confusion_matrix(post_labels, pred_binary)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'fpr': fpr,
                'tpr': tpr,
                'confusion_matrix': cm,
                'predictions': predictions,
                'labels': post_labels
            }
    
    def evaluate_early_detection(self, model, test_data):
        """
        Evaluate model for early detection of misinformation.
        
        Args:
            model (nn.Module): Model to evaluate.
            test_data (dict): Test data.
            
        Returns:
            dict: Early detection metrics.
        """
        model.eval()
        
        # Extract posts, users, and timestamps
        post_mask = test_data['post_mask'].cpu().numpy()
        post_indices = np.where(post_mask)[0]
        
        # Get post timestamps (first interaction with each post)
        post_timestamps = {}
        post_interaction_times = {}
        post_predictions = {}
        
        # For each edge, check if it connects to a post
        edge_index = test_data['edge_index'].cpu().numpy()
        edge_times = test_data['edge_times'].cpu().numpy()
        
        # Get timestamps for each post's interactions
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            time = edge_times[i]
            
            # Check if either source or destination is a post
            if src in post_indices:
                post_id = src
                if post_id not in post_interaction_times:
                    post_interaction_times[post_id] = []
                post_interaction_times[post_id].append(time)
            
            if dst in post_indices:
                post_id = dst
                if post_id not in post_interaction_times:
                    post_interaction_times[post_id] = []
                post_interaction_times[post_id].append(time)
        
        # Sort interactions by time for each post
        for post_id, times in post_interaction_times.items():
            sorted_times = sorted(times)
            post_timestamps[post_id] = sorted_times
        
        # Evaluate model at each interaction time for each post
        with torch.no_grad():
            for post_id, times in post_timestamps.items():
                post_predictions[post_id] = []
                
                for i, time in enumerate(times):
                    # Create graph with edges up to this time
                    time_mask = edge_times <= time
                    edges_to_include = np.where(time_mask)[0]
                    
                    # Create a subgraph with edges up to this time
                    temp_data = {
                        'node_features': test_data['node_features'],
                        'edge_index': test_data['edge_index'][:, edges_to_include],
                        'edge_attr': test_data['edge_attr'][edges_to_include],
                        'edge_times': test_data['edge_times'][edges_to_include],
                        'labels': test_data['labels'],
                        'post_mask': test_data['post_mask'],
                        'node_types': test_data['node_types'],
                        'edge_types': test_data['edge_types'][edges_to_include]
                    }
                    
                    # Add snapshots if needed (for GCN)
                    if 'snapshots' in test_data:
                        temp_data['snapshots'] = [
                            {
                                'edge_index': snapshot['edge_index'],
                                'edge_attr': snapshot['edge_attr'],
                                'edge_types': snapshot['edge_types'],
                                'time_window': snapshot['time_window']
                            } for snapshot in test_data['snapshots'] 
                            if snapshot['time_window'][1] <= time
                        ]
                    
                    # Move data to device
                    model_data = self.prepare_data_for_model(temp_data)
                    
                    # Get model prediction for this post
                    predictions = model(model_data)
                    post_pred = predictions[post_mask][post_indices == post_id].cpu().numpy()[0]
                    post_predictions[post_id].append(post_pred)
        
        # Convert timestamps to datetime objects for time difference calculation
        post_datetimes = {}
        for post_id, times in post_timestamps.items():
            # Convert Unix timestamps to datetime objects
            post_datetimes[post_id] = [datetime.fromtimestamp(t) for t in times]
        
        # Get labels for all posts
        post_labels = test_data['labels'][post_indices].cpu().numpy()
        
        # Create lists in the format needed for early detection metrics
        prediction_lists = []
        label_lists = []
        datetime_lists = []
        
        for i, post_id in enumerate(post_indices):
            if post_id in post_predictions:
                prediction_lists.append(post_predictions[post_id])
                label_lists.append(post_labels[i])
                datetime_lists.append(post_datetimes[post_id])
        
        # Calculate early detection metrics
        early_metrics = {}
        
        # Time to detection
        early_metrics['time_to_detection'] = EarlyDetectionMetrics.time_to_detection(
            prediction_lists, label_lists, datetime_lists
        )
        
        # Early precision
        early_metrics['early_precision'] = EarlyDetectionMetrics.early_precision(
            prediction_lists, label_lists, datetime_lists, self.time_windows
        )
        
        # Early recall
        early_metrics['early_recall'] = EarlyDetectionMetrics.early_recall(
            prediction_lists, label_lists, datetime_lists, self.time_windows
        )
        
        # Early F1
        early_metrics['early_f1'] = EarlyDetectionMetrics.early_f1(
            early_metrics['early_precision'], early_metrics['early_recall']
        )
        
        return early_metrics
    
    def plot_roc_curve(self, model_results, dataset_name):
        """
        Plot ROC curves for all models.
        
        Args:
            model_results (dict): Results for each model.
            dataset_name (str): Name of the dataset.
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, results in model_results.items():
            plt.plot(
                results['fpr'], 
                results['tpr'],
                label=f"{model_name} (AUC = {results['roc_auc']:.3f})"
            )
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {dataset_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f'roc_curve_{dataset_name}.png'), dpi=300)
        plt.close()
    
    def plot_confusion_matrices(self, model_results, dataset_name):
        """
        Plot confusion matrices for all models.
        
        Args:
            model_results (dict): Results for each model.
            dataset_name (str): Name of the dataset.
        """
        num_models = len(model_results)
        fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5))
        
        # If only one model, axes won't be an array
        if num_models == 1:
            axes = [axes]
        
        for ax, (model_name, results) in zip(axes, model_results.items()):
            cm = results['confusion_matrix']
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot confusion matrix
            sns.heatmap(
                cm_normalized, 
                annot=cm,
                fmt='d', 
                cmap='Blues', 
                ax=ax,
                cbar=False,
                xticklabels=['Not Misinfo', 'Misinfo'],
                yticklabels=['Not Misinfo', 'Misinfo']
            )
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'{model_name}')
        
        plt.tight_layout()
        plt.suptitle(f'Confusion Matrices - {dataset_name}', y=1.05)
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f'confusion_matrices_{dataset_name}.png'), dpi=300)
        plt.close()
    
    def plot_early_detection_comparison(self, model_early_results, dataset_name):
        """
        Plot early detection performance comparison.
        
        Args:
            model_early_results (dict): Early detection results for each model.
            dataset_name (str): Name of the dataset.
        """
        # Plot early F1 score over time windows
        plt.figure(figsize=(12, 8))
        
        for model_name, results in model_early_results.items():
            windows = list(results['early_f1'].keys())
            f1_values = list(results['early_f1'].values())
            
            plt.plot(windows, f1_values, marker='o', label=model_name)
        
        plt.xlabel('Time Window (hours)')
        plt.ylabel('F1 Score')
        plt.title(f'Early Detection F1 Score - {dataset_name}')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f'early_f1_{dataset_name}.png'), dpi=300)
        plt.close()
        
        # Plot time to detection
        times_to_detection = {
            model_name: results['time_to_detection'] 
            for model_name, results in model_early_results.items()
        }
        
        plt.figure(figsize=(10, 6))
        
        # Filter out inf values for plotting
        filtered_times = {
            model: time for model, time in times_to_detection.items() 
            if time != float('inf')
        }
        
        if filtered_times:
            model_names = list(filtered_times.keys())
            detection_times = list(filtered_times.values())
            
            bars = plt.bar(model_names, detection_times)
            
            # Add value labels above bars
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 0.1,
                    f'{height:.2f}',
                    ha='center', va='bottom'
                )
            
            plt.ylabel('Average Time to Detection (hours)')
            plt.title(f'Time to Detection Comparison - {dataset_name}')
            plt.grid(axis='y')
            
            # Save plot
            plt.savefig(os.path.join(self.output_dir, f'time_to_detection_{dataset_name}.png'), dpi=300)
            plt.close()
    
    def plot_metric_comparison(self, model_results, dataset_name):
        """
        Plot standard metric comparison.
        
        Args:
            model_results (dict): Results for each model.
            dataset_name (str): Name of the dataset.
        """
        # Extract metrics for comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        model_names = list(model_results.keys())
        
        # Create a DataFrame for plotting
        data = []
        for model_name, results in model_results.items():
            for metric in metrics:
                data.append({
                    'Model': model_name,
                    'Metric': metric.capitalize(),
                    'Value': results[metric]
                })
        
        df = pd.DataFrame(data)
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        sns.barplot(x='Metric', y='Value', hue='Model', data=df)
        
        plt.title(f'Performance Metrics Comparison - {dataset_name}')
        plt.ylim(0, 1)
        plt.grid(axis='y')
        plt.legend(title='Model')
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f'metrics_comparison_{dataset_name}.png'), dpi=300)
        plt.close()
    
    def evaluate_models(self, models, dataset_name):
        """
        Evaluate and compare multiple models on a dataset.
        
        Args:
            models (list): List of model names to evaluate.
            dataset_name (str): Name of the dataset to use.
            
        Returns:
            tuple: (standard_metrics, early_detection_metrics)
        """
        # Load dataset
        graph_data = self.load_data(dataset_name)
        
        # Split data
        _, _, test_data = self.split_data_temporal(graph_data)
        
        standard_results = {}
        early_detection_results = {}
        
        for model_name in models:
            print(f"Evaluating {model_name} on {dataset_name}...")
            
            # Load model
            model = self.load_model(model_name, dataset_name, graph_data)
            
            # Evaluate standard metrics
            standard_results[model_name] = self.evaluate_model(model, test_data)
            
            # Evaluate early detection metrics
            early_detection_results[model_name] = self.evaluate_early_detection(model, test_data)
        
        # Generate plots
        print("Generating plots...")
        self.plot_roc_curve(standard_results, dataset_name)
        self.plot_confusion_matrices(standard_results, dataset_name)
        self.plot_early_detection_comparison(early_detection_results, dataset_name)
        self.plot_metric_comparison(standard_results, dataset_name)
        
        # Save results to file
        results_path = os.path.join(
            self.config['general']['save_dir'],
            f'evaluation_results_{dataset_name}.json'
        )
        
        # Convert numpy arrays and any non-serializable objects to lists
        serializable_results = {}
        for model_name, results in standard_results.items():
            serializable_results[model_name] = {
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1': float(results['f1']),
                'roc_auc': float(results['roc_auc']),
                # Don't include large arrays in the JSON
                'confusion_matrix': results['confusion_matrix'].tolist()
            }
        
        # Create a combined results dictionary
        combined_results = {
            'standard_metrics': serializable_results,
            'early_detection': early_detection_results
        }
        
        with open(results_path, 'w') as f:
            json.dump(combined_results, f, indent=4)
        
        print(f"Evaluation results saved to {results_path}")
        
        return standard_results, early_detection_results
    
    def run_evaluation(self, models=None, datasets=None):
        """
        Run evaluation for specified models and datasets.
        
        Args:
            models (list, optional): List of models to evaluate. If None, uses all models in config.
            datasets (list, optional): List of datasets to use. If None, uses all datasets in config.
        """
        if models is None:
            models = list(self.config['models'].keys())
        
        if datasets is None:
            datasets = self.config['data']['datasets']
        
        for dataset_name in datasets:
            print(f"\n{'='*50}")
            print(f"Evaluating models on {dataset_name}")
            print(f"{'='*50}\n")
            
            self.evaluate_models(models, dataset_name)
        
        print("\nEvaluation complete!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate misinformation detection models")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--models", type=str, nargs="+", 
                       help="Models to evaluate (space-separated)")
    parser.add_argument("--datasets", type=str, nargs="+",
                       help="Datasets to use (space-separated)")
    args = parser.parse_args()
    
    evaluator = Evaluator(config_path=args.config)
    evaluator.run_evaluation(models=args.models, datasets=args.datasets)
