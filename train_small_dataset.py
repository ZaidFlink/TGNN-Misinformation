#!/usr/bin/env python
"""
Simplified training script for the small CoAID dataset.
This script adapts the models and training procedure to work with a very small dataset.
"""

import os
import pickle
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

# Function to load the processed data
def load_data(dataset_name="coaid"):
    data_path = f'data/processed/{dataset_name}_processed.pkl'
    with open(data_path, 'rb') as f:
        graph_data = pickle.load(f)
    return graph_data

# Function to create temporal splits
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

# Simplified TGN model for small datasets
class SimplifiedTGN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=32, output_dim=1):
        super(SimplifiedTGN, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # Node encoder
        self.node_encoder = torch.nn.Linear(node_dim, hidden_dim)
        
        # Edge encoder
        self.edge_encoder = torch.nn.Linear(edge_dim, hidden_dim)
        
        # Graph convolution layer
        self.conv = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Output layer
        self.output = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, node_features, edge_index, edge_attr, post_mask):
        # Encode node features
        node_embeddings = torch.relu(self.node_encoder(node_features))
        
        # Create a simple message-passing mechanism
        src, dst = edge_index
        edge_embeddings = torch.relu(self.edge_encoder(edge_attr))
        
        # Aggregate messages for each node
        messages = torch.zeros_like(node_embeddings)
        for i in range(edge_index.shape[1]):
            source = src[i]
            target = dst[i]
            message = torch.cat([node_embeddings[source], edge_embeddings[i]], dim=0)
            messages[target] += self.conv(message.unsqueeze(0)).squeeze(0)
        
        # Update node embeddings
        node_embeddings = node_embeddings + messages
        
        # Only predict for post nodes
        post_embeddings = node_embeddings[post_mask]
        
        # Output predictions
        logits = self.output(post_embeddings).squeeze(-1)
        return torch.sigmoid(logits)

# Simplified static GCN model
class SimplifiedGCN(torch.nn.Module):
    def __init__(self, node_dim, hidden_dim=32, output_dim=1):
        super(SimplifiedGCN, self).__init__()
        self.node_dim = node_dim
        
        # Node encoder
        self.encoder = torch.nn.Linear(node_dim, hidden_dim)
        
        # Graph convolution layers
        self.conv1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.conv2 = torch.nn.Linear(hidden_dim, hidden_dim)
        
        # Output layer
        self.output = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, node_features, edge_index, post_mask):
        # Encode node features
        node_embeddings = torch.relu(self.encoder(node_features))
        
        # Simple graph convolution
        src, dst = edge_index
        
        # First convolution
        aggregated = torch.zeros_like(node_embeddings)
        for i in range(edge_index.shape[1]):
            aggregated[dst[i]] += node_embeddings[src[i]]
        
        node_embeddings = torch.relu(self.conv1(aggregated + node_embeddings))
        
        # Second convolution
        aggregated = torch.zeros_like(node_embeddings)
        for i in range(edge_index.shape[1]):
            aggregated[dst[i]] += node_embeddings[src[i]]
        
        node_embeddings = torch.relu(self.conv2(aggregated + node_embeddings))
        
        # Only predict for post nodes
        post_embeddings = node_embeddings[post_mask]
        
        # Output predictions
        logits = self.output(post_embeddings).squeeze(-1)
        return torch.sigmoid(logits)

def train_model(model, graph_data, train_mask, val_mask, test_mask, 
                model_type='tgn', epochs=50, lr=0.001, weight_decay=0.0001):
    """Train a model on the graph data."""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Get data
    node_features = graph_data['node_features'].to(device)
    edge_index = graph_data['edge_index'].to(device)
    edge_attr = graph_data['edge_attr'].to(device)
    post_mask = graph_data['post_mask'].to(device)
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
    
    # Loss function
    loss_fn = torch.nn.BCELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    test_metrics = {'accuracy': [], 'f1': [], 'auc': [], 'precision': [], 'recall': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        if model_type == 'tgn':
            outputs = model(node_features, train_edge_index, train_edge_attr, post_mask)
        else:  # GCN
            outputs = model(node_features, train_edge_index, post_mask)
        
        loss = loss_fn(outputs, labels.float())
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            if model_type == 'tgn':
                val_outputs = model(node_features, val_edge_index, val_edge_attr, post_mask)
            else:  # GCN
                val_outputs = model(node_features, val_edge_index, post_mask)
            
            val_loss = loss_fn(val_outputs, labels.float())
            val_losses.append(val_loss.item())
            
            # Test metrics
            if model_type == 'tgn':
                test_outputs = model(node_features, test_edge_index, test_edge_attr, post_mask)
            else:  # GCN
                test_outputs = model(node_features, test_edge_index, post_mask)
            
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
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
            print(f"Test Metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        if model_type == 'tgn':
            test_outputs = model(node_features, test_edge_index, test_edge_attr, post_mask)
        else:  # GCN
            test_outputs = model(node_features, test_edge_index, post_mask)
        
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
    
    print("\nFinal Test Metrics:")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"F1 Score: {final_f1:.4f}")
    print(f"AUC: {final_auc:.4f}")
    print(f"Precision: {final_precision:.4f}")
    print(f"Recall: {final_recall:.4f}")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {model_type.upper()}')
    plt.legend()
    plt.savefig(f'{model_type}_loss.png')
    
    # Plot test metrics
    plt.figure(figsize=(12, 8))
    for metric in test_metrics:
        plt.plot(test_metrics[metric], label=metric.capitalize())
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title(f'Test Metrics - {model_type.upper()}')
    plt.legend()
    plt.savefig(f'{model_type}_metrics.png')
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_metrics': test_metrics,
        'final_metrics': {
            'accuracy': final_accuracy,
            'f1': final_f1,
            'auc': final_auc,
            'precision': final_precision,
            'recall': final_recall
        }
    }

def main():
    # Load data
    print("Loading CoAID processed data...")
    graph_data = load_data()
    
    # Create temporal splits
    train_mask, val_mask, test_mask = create_temporal_splits(graph_data)
    
    # Get dimensions
    node_dim = graph_data['node_features'].shape[1]
    edge_dim = graph_data['edge_attr'].shape[1]
    
    print(f"Node feature dimension: {node_dim}")
    print(f"Edge feature dimension: {edge_dim}")
    
    # Train simplified TGN
    print("\n--- Training Simplified TGN ---")
    tgn_model = SimplifiedTGN(node_dim, edge_dim)
    tgn_results = train_model(tgn_model, graph_data, train_mask, val_mask, test_mask, model_type='tgn')
    
    # Train simplified GCN
    print("\n--- Training Simplified GCN ---")
    gcn_model = SimplifiedGCN(node_dim)
    gcn_results = train_model(gcn_model, graph_data, train_mask, val_mask, test_mask, model_type='gcn')
    
    # Compare results
    print("\n=== Model Comparison ===")
    print("TGN vs GCN:")
    print(f"Accuracy: {tgn_results['final_metrics']['accuracy']:.4f} vs {gcn_results['final_metrics']['accuracy']:.4f}")
    print(f"F1 Score: {tgn_results['final_metrics']['f1']:.4f} vs {gcn_results['final_metrics']['f1']:.4f}")
    print(f"AUC: {tgn_results['final_metrics']['auc']:.4f} vs {gcn_results['final_metrics']['auc']:.4f}")
    print(f"Precision: {tgn_results['final_metrics']['precision']:.4f} vs {gcn_results['final_metrics']['precision']:.4f}")
    print(f"Recall: {tgn_results['final_metrics']['recall']:.4f} vs {gcn_results['final_metrics']['recall']:.4f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    # Plot comparison
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
    plt.title('TGN vs GCN Performance Comparison')
    plt.xticks(x, [m.capitalize() for m in metrics])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    
    print("\nResults saved. Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main() 