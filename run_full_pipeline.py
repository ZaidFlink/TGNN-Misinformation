#!/usr/bin/env python
"""
Full Pipeline Script for Misinformation Detection

This script runs the complete pipeline for temporal graph-based misinformation detection:
1. Loads the processed dataset
2. Trains the models (TGN, GCN)
3. Evaluates performance including early detection capability
"""

import os
import pickle
import numpy as np
import torch
import yaml
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# Load configuration
with open('configs/config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() and CONFIG['general']['device'] == 'cuda' else 'cpu')
print(f"Using device: {device}")

class TGN(torch.nn.Module):
    """Simplified Temporal Graph Network for misinformation detection."""
    
    def __init__(self, node_dim, edge_dim, hidden_dim=64, output_dim=1):
        super(TGN, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # Node encoder
        self.node_encoder = torch.nn.Linear(node_dim, hidden_dim)
        
        # Edge encoder
        self.edge_encoder = torch.nn.Linear(edge_dim, hidden_dim)
        
        # Message passing
        self.message_aggregator = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Update function
        self.update_function = torch.nn.GRUCell(hidden_dim, hidden_dim)
        
        # Output layer
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, node_features, edge_index, edge_attr, post_mask, n_layers=2):
        # Initialize node embeddings
        node_embeddings = torch.relu(self.node_encoder(node_features))
        
        # Encode edge features
        edge_embeddings = torch.relu(self.edge_encoder(edge_attr))
        
        # Message passing for n_layers
        for _ in range(n_layers):
            # Initialize messages
            messages = torch.zeros_like(node_embeddings)
            
            # Aggregate messages
            src, dst = edge_index
            for i in range(edge_index.shape[1]):
                source = src[i]
                target = dst[i]
                
                # Combine node and edge embeddings as the message
                msg = torch.cat([node_embeddings[source], edge_embeddings[i]], dim=0)
                msg = self.message_aggregator(msg.unsqueeze(0)).squeeze(0)
                
                # Aggregate messages for each node
                messages[target] += msg
            
            # Update node embeddings (memory)
            node_embeddings = self.update_function(messages, node_embeddings)
        
        # Apply prediction only to post nodes
        post_embeddings = node_embeddings[post_mask]
        
        # Predict
        logits = self.output_layer(post_embeddings).squeeze(-1)
        
        return torch.sigmoid(logits)


class GCN(torch.nn.Module):
    """Static Graph Convolutional Network for misinformation detection."""
    
    def __init__(self, node_dim, hidden_dim=64, output_dim=1):
        super(GCN, self).__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        
        # Node encoder
        self.node_encoder = torch.nn.Linear(node_dim, hidden_dim)
        
        # Graph convolution layers
        self.conv1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.conv2 = torch.nn.Linear(hidden_dim, hidden_dim)
        
        # Output layer
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, node_features, edge_index, post_mask):
        # Initialize node embeddings
        node_embeddings = torch.relu(self.node_encoder(node_features))
        
        # Simple graph convolution (aggregation)
        src, dst = edge_index
        
        # First convolution
        aggregated = torch.zeros_like(node_embeddings)
        for i in range(edge_index.shape[1]):
            aggregated[dst[i]] += node_embeddings[src[i]]
        
        # Update embeddings
        node_embeddings = torch.relu(self.conv1(aggregated + node_embeddings))
        
        # Second convolution
        aggregated = torch.zeros_like(node_embeddings)
        for i in range(edge_index.shape[1]):
            aggregated[dst[i]] += node_embeddings[src[i]]
        
        # Update embeddings
        node_embeddings = torch.relu(self.conv2(aggregated + node_embeddings))
        
        # Apply prediction only to post nodes
        post_embeddings = node_embeddings[post_mask]
        
        # Predict
        logits = self.output_layer(post_embeddings).squeeze(-1)
        
        return torch.sigmoid(logits)


def create_temporal_splits(graph_data, train_ratio=0.7, val_ratio=0.15):
    """Create training, validation, and test splits based on temporal ordering."""
    # Get temporal ordering of edges
    edge_times = graph_data['edge_times'].numpy()
    sorted_indices = np.argsort(edge_times)
    
    # Calculate split indices
    n_edges = len(edge_times)
    train_idx = int(n_edges * train_ratio)
    val_idx = int(n_edges * (train_ratio + val_ratio))
    
    # Create masks
    train_mask = sorted_indices[:train_idx]
    val_mask = sorted_indices[train_idx:val_idx]
    test_mask = sorted_indices[val_idx:]
    
    print(f"Created temporal splits: {len(train_mask)} train edges, {len(val_mask)} validation edges, {len(test_mask)} test edges.")
    
    return train_mask, val_mask, test_mask


def train_model(model, graph_data, train_mask, val_mask, test_mask, 
                model_type='tgn', epochs=50, lr=0.001, weight_decay=0.0001,
                batch_size=4096):
    """Train a model on the graph data."""
    # Setup device
    model.to(device)
    
    # Get data
    node_features = graph_data['node_features'].to(device)
    edge_index = graph_data['edge_index'].to(device)
    edge_attr = graph_data['edge_attr'].to(device)
    post_mask = graph_data['post_mask'].to(device)
    
    # Get labels (only for post nodes)
    post_indices = torch.where(post_mask)[0]
    labels = graph_data['labels'][post_mask].to(device)
    
    # Filter edges by mask for each split
    train_edge_index = edge_index[:, train_mask]
    train_edge_attr = edge_attr[train_mask]
    
    val_edge_index = edge_index[:, val_mask]
    val_edge_attr = edge_attr[val_mask]
    
    test_edge_index = edge_index[:, test_mask]
    test_edge_attr = edge_attr[test_mask]
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Loss function - use weighted BCE if class imbalance
    if CONFIG['data']['class_weight']['enable']:
        # Calculate weights based on class distribution
        pos_weight = (labels == 0).sum().float() / (labels == 1).sum().float()
        loss_fn = torch.nn.BCELoss(weight=torch.tensor([pos_weight]).to(device))
    else:
        loss_fn = torch.nn.BCELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    test_metrics = {'accuracy': [], 'f1': [], 'auc': [], 'precision': [], 'recall': []}
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    # For large graphs, process in batches
    def process_edges_in_batches(edges, edge_attrs=None, batch_size=batch_size):
        total_batches = (edges.shape[1] + batch_size - 1) // batch_size
        
        all_outputs = []
        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, edges.shape[1])
            
            batch_edges = edges[:, start_idx:end_idx]
            
            if edge_attrs is not None and model_type == 'tgn':
                batch_attrs = edge_attrs[start_idx:end_idx]
                outputs = model(node_features, batch_edges, batch_attrs, post_mask)
            else:
                outputs = model(node_features, batch_edges, post_mask)
                
            all_outputs.append(outputs)
            
        # Combine outputs by taking the mean
        return torch.mean(torch.stack([out for out in all_outputs]), dim=0)
    
    print(f"Starting training {model_type} for {epochs} epochs")
    start_time = time.time()
    
    for epoch in tqdm(range(epochs)):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Process edges in batches for large graphs
        outputs = process_edges_in_batches(train_edge_index, train_edge_attr)
        
        loss = loss_fn(outputs, labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['training']['gradient_clipping'])
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            # Process edges in batches for large graphs
            val_outputs = process_edges_in_batches(val_edge_index, val_edge_attr)
            
            val_loss = loss_fn(val_outputs, labels.float())
            val_losses.append(val_loss.item())
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            # Test metrics
            test_outputs = process_edges_in_batches(test_edge_index, test_edge_attr)
            
            test_preds = (test_outputs > 0.5).float().cpu().numpy()
            test_labels = labels.cpu().numpy()
            test_probs = test_outputs.cpu().numpy()
            
            # Calculate metrics
            accuracy = accuracy_score(test_labels, test_preds)
            f1 = f1_score(test_labels, test_preds, zero_division=0)
            
            # Handle case where only one class is present
            if len(np.unique(test_labels)) > 1:
                auc = roc_auc_score(test_labels, test_probs)
            else:
                auc = 0.5  # Default for binary classification with one class
                
            precision = precision_score(test_labels, test_preds, zero_division=0)
            recall = recall_score(test_labels, test_preds, zero_division=0)
            
            test_metrics['accuracy'].append(accuracy)
            test_metrics['f1'].append(f1)
            test_metrics['auc'].append(auc)
            test_metrics['precision'].append(precision)
            test_metrics['recall'].append(recall)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
            print(f"Test Metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        # Early stopping
        if early_stop_counter >= CONFIG['general']['early_stopping_patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = process_edges_in_batches(test_edge_index, test_edge_attr)
        
        test_preds = (test_outputs > 0.5).float().cpu().numpy()
        test_labels = labels.cpu().numpy()
        test_probs = test_outputs.cpu().numpy()
        
        # Calculate final metrics
        final_accuracy = accuracy_score(test_labels, test_preds)
        final_f1 = f1_score(test_labels, test_preds, zero_division=0)
        
        if len(np.unique(test_labels)) > 1:
            final_auc = roc_auc_score(test_labels, test_probs)
        else:
            final_auc = 0.5
            
        final_precision = precision_score(test_labels, test_preds, zero_division=0)
        final_recall = recall_score(test_labels, test_preds, zero_division=0)
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    print("\nFinal Test Metrics:")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"F1 Score: {final_f1:.4f}")
    print(f"AUC: {final_auc:.4f}")
    print(f"Precision: {final_precision:.4f}")
    print(f"Recall: {final_recall:.4f}")
    
    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), f'checkpoints/{model_type}_model.pt')
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {model_type.upper()}')
    plt.legend()
    plt.savefig(f'results/{model_type}_loss.png')
    
    # Plot test metrics
    plt.figure(figsize=(12, 8))
    for metric in test_metrics:
        plt.plot(test_metrics[metric], label=metric.capitalize())
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title(f'Test Metrics - {model_type.upper()}')
    plt.legend()
    plt.savefig(f'results/{model_type}_metrics.png')
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_metrics': test_metrics,
        'final_metrics': {
            'accuracy': final_accuracy,
            'f1': final_f1,
            'auc': final_auc,
            'precision': final_precision,
            'recall': final_recall,
            'training_time': training_time
        }
    }


def analyze_early_detection(models_results, graph_data, dataset_name):
    """Analyze early detection capabilities of models."""
    print(f"\nAnalyzing early detection capabilities for {dataset_name}...")
    
    # Get the times associated with each edge
    edge_times = graph_data['edge_times'].numpy()
    edge_index = graph_data['edge_index']
    post_mask = graph_data['post_mask']
    labels = graph_data['labels'][post_mask].numpy()
    
    # Get post node IDs
    post_ids = torch.where(post_mask)[0].tolist()
    
    # Dictionary to track first interaction time for each post
    first_interaction_times = {}
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i].numpy()
        time = edge_times[i]
        
        # If this edge connects to a post node, record the time
        if src in post_ids:
            if src not in first_interaction_times:
                first_interaction_times[src] = time
            else:
                first_interaction_times[src] = min(first_interaction_times[src], time)
        
        if dst in post_ids:
            if dst not in first_interaction_times:
                first_interaction_times[dst] = time
            else:
                first_interaction_times[dst] = min(first_interaction_times[dst], time)
    
    # Create time windows for analysis
    min_time = min(edge_times)
    max_time = max(edge_times)
    total_duration = max_time - min_time
    
    num_windows = 10
    time_windows = []
    for i in range(num_windows):
        window_start = min_time + i * (total_duration / num_windows)
        window_end = min_time + (i + 1) * (total_duration / num_windows)
        time_windows.append((window_start, window_end))
    
    # Analyze early detection
    results = {}
    
    for model_name, model_result in models_results.items():
        model = model_result['model']
        model.eval()
        
        detection_results = []
        
        # For each time window, evaluate model's detection capability
        for window_idx, (start_time, end_time) in enumerate(time_windows):
            # Get edges within this time window
            mask = (edge_times >= min_time) & (edge_times <= end_time)
            window_edge_index = edge_index[:, mask].to(device)
            
            if model_name == 'tgn':
                window_edge_attr = graph_data['edge_attr'][mask].to(device)
            
            # Skip if no edges
            if window_edge_index.shape[1] == 0:
                continue
            
            # Predict
            with torch.no_grad():
                if model_name == 'tgn':
                    predictions = model(
                        graph_data['node_features'].to(device),
                        window_edge_index,
                        window_edge_attr,
                        graph_data['post_mask'].to(device)
                    )
                else:  # GCN
                    predictions = model(
                        graph_data['node_features'].to(device),
                        window_edge_index,
                        graph_data['post_mask'].to(device)
                    )
            
            # Calculate metrics for misinformation posts
            predictions = predictions.cpu().numpy()
            pred_labels = (predictions > 0.5).astype(int)
            
            # Convert times to dates for display
            start_date = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M')
            end_date = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M')
            
            # For each misinformation post, check if detected correctly
            for i, post_id in enumerate(post_ids):
                # Only analyze misinformation posts
                if labels[i] != 1:
                    continue
                    
                # If this post had interactions by this time window
                if post_id in first_interaction_times and first_interaction_times[post_id] <= end_time:
                    time_since_first = end_time - first_interaction_times[post_id]
                    correctly_detected = pred_labels[i] == 1
                    
                    # Store results
                    detection_results.append({
                        'post_id': post_id,
                        'window_idx': window_idx,
                        'time_window': f"{start_date} - {end_date}",
                        'time_since_first_interaction': time_since_first,
                        'time_since_first_hours': time_since_first / 3600,
                        'correctly_detected': correctly_detected,
                        'confidence': predictions[i]
                    })
        
        results[model_name] = pd.DataFrame(detection_results)
    
    # Compare early detection results
    comparison = {}
    
    for model_name, df in results.items():
        if len(df) == 0:
            print(f"No detection results for {model_name}")
            continue
            
        # Calculate detection rate over time
        detection_rate = df.groupby('window_idx')['correctly_detected'].mean()
        
        # Calculate time to detection
        correct_detections = df[df['correctly_detected']]
        if len(correct_detections) > 0:
            time_to_detection = correct_detections.groupby('post_id')['time_since_first_hours'].min().mean()
        else:
            time_to_detection = float('nan')
        
        comparison[model_name] = {
            'detection_rate': detection_rate,
            'final_detection_rate': detection_rate.iloc[-1] if len(detection_rate) > 0 else 0,
            'time_to_detection': time_to_detection
        }
    
    # Visualize early detection comparison
    if len(comparison) >= 2:
        # Plot detection rate over time
        plt.figure(figsize=(12, 6))
        
        for model_name, metrics in comparison.items():
            if 'detection_rate' in metrics and len(metrics['detection_rate']) > 0:
                plt.plot(metrics['detection_rate'].index, metrics['detection_rate'].values, 
                         'o-', linewidth=2, label=model_name.upper())
        
        plt.xlabel('Time Window')
        plt.ylabel('Detection Rate')
        plt.title(f'Misinformation Detection Rate Over Time - {dataset_name}')
        plt.ylim(0, 1.05)
        plt.legend()
        plt.savefig(f'results/{dataset_name}_detection_rate.png')
        
        # Plot time to detection comparison
        plt.figure(figsize=(10, 6))
        model_names = []
        detection_times = []
        
        for model_name, metrics in comparison.items():
            if 'time_to_detection' in metrics and not np.isnan(metrics['time_to_detection']):
                model_names.append(model_name.upper())
                detection_times.append(metrics['time_to_detection'])
        
        if detection_times:
            bars = plt.bar(model_names, detection_times)
            plt.ylabel('Average Time to Detection (hours)')
            plt.title(f'Average Time to Correct Misinformation Detection - {dataset_name}')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f} h', ha='center', va='bottom')
            
            plt.savefig(f'results/{dataset_name}_time_to_detection.png')
    
    return comparison


def run_pipeline(dataset_name):
    """Run the full pipeline for a dataset."""
    print(f"Running pipeline for {dataset_name}...")
    
    # Load the processed data
    data_path = f'data/processed/{dataset_name}_processed.pkl'
    if not os.path.exists(data_path):
        print(f"Error: Processed data not found at {data_path}")
        return
    
    with open(data_path, 'rb') as f:
        graph_data = pickle.load(f)
    
    print(f"Loaded {dataset_name} graph with {graph_data['num_nodes']} nodes and {graph_data['edge_index'].shape[1]} edges")
    
    # Create temporal splits
    train_mask, val_mask, test_mask = create_temporal_splits(graph_data)
    
    # Get dimensions
    node_dim = graph_data['node_features'].shape[1]
    edge_dim = graph_data['edge_attr'].shape[1]
    
    print(f"Node feature dimension: {node_dim}")
    print(f"Edge feature dimension: {edge_dim}")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Train TGN model
    print("\n--- Training TGN ---")
    tgn_model = TGN(node_dim, edge_dim)
    tgn_results = train_model(
        tgn_model, graph_data, train_mask, val_mask, test_mask, model_type='tgn',
        epochs=CONFIG['general']['num_epochs'], 
        lr=CONFIG['training']['learning_rate'],
        weight_decay=CONFIG['training']['weight_decay']
    )
    
    # Train GCN model
    print("\n--- Training GCN ---")
    gcn_model = GCN(node_dim)
    gcn_results = train_model(
        gcn_model, graph_data, train_mask, val_mask, test_mask, model_type='gcn',
        epochs=CONFIG['general']['num_epochs'],
        lr=CONFIG['training']['learning_rate'],
        weight_decay=CONFIG['training']['weight_decay']
    )
    
    # Compare results
    print("\n=== Model Comparison ===")
    print("TGN vs GCN:")
    print(f"Accuracy: {tgn_results['final_metrics']['accuracy']:.4f} vs {gcn_results['final_metrics']['accuracy']:.4f}")
    print(f"F1 Score: {tgn_results['final_metrics']['f1']:.4f} vs {gcn_results['final_metrics']['f1']:.4f}")
    print(f"AUC: {tgn_results['final_metrics']['auc']:.4f} vs {gcn_results['final_metrics']['auc']:.4f}")
    print(f"Precision: {tgn_results['final_metrics']['precision']:.4f} vs {gcn_results['final_metrics']['precision']:.4f}")
    print(f"Recall: {tgn_results['final_metrics']['recall']:.4f} vs {gcn_results['final_metrics']['recall']:.4f}")
    print(f"Training Time: {tgn_results['final_metrics']['training_time']:.2f}s vs {gcn_results['final_metrics']['training_time']:.2f}s")
    
    # Plot comparison of metrics
    metrics = ['accuracy', 'f1', 'auc', 'precision', 'recall']
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    tgn_values = [tgn_results['final_metrics'][m] for m in metrics]
    gcn_values = [gcn_results['final_metrics'][m] for m in metrics]
    
    plt.bar(x - width/2, tgn_values, width, label='TGN')
    plt.bar(x + width/2, gcn_values, width, label='GCN')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(f'TGN vs GCN Performance Comparison - {dataset_name}')
    plt.xticks(x, [m.capitalize() for m in metrics])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/{dataset_name}_model_comparison.png')
    
    # Analyze early detection
    models_results = {
        'tgn': tgn_results,
        'gcn': gcn_results
    }
    
    early_detection_comparison = analyze_early_detection(models_results, graph_data, dataset_name)
    
    # Save results summary
    summary = {
        'dataset': dataset_name,
        'tgn_metrics': tgn_results['final_metrics'],
        'gcn_metrics': gcn_results['final_metrics'],
        'early_detection': early_detection_comparison
    }
    
    with open(f'results/{dataset_name}_summary.pkl', 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"\nResults saved for {dataset_name}. Check the generated PNG files for visualizations.")
    
    return summary


def main():
    """Run the full pipeline for all datasets."""
    print("Starting misinformation detection pipeline...")
    
    # Process and run on each dataset
    results = {}
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # FakeNewsNet dataset
    if os.path.exists('data/processed/fakenewsnet_processed.pkl'):
        print("\n========================")
        print("PROCESSING FAKENEWSNET")
        print("========================\n")
        results['fakenewsnet'] = run_pipeline('fakenewsnet')
    
    # CoAID dataset - skip since it didn't contain data in our structure
    # results['coaid'] = run_pipeline('coaid')
    
    print("\nPipeline completed successfully!")
    print("Check the 'results' directory for visualizations and performance metrics.")


if __name__ == "__main__":
    main() 