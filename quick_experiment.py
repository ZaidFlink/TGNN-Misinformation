#!/usr/bin/env python
"""
Quick Experiment Script

This script runs a simplified experiment on a subset of the data
to quickly demonstrate temporal vs. static model performance.
"""

import os
import pickle
import numpy as np
import torch
import yaml
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load configuration
with open('configs/config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Simplified TGN model
class SimpleTGN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=32):
        super(SimpleTGN, self).__init__()
        self.node_encoder = torch.nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = torch.nn.Linear(edge_dim, hidden_dim)
        self.message_layer = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.update_layer = torch.nn.GRUCell(hidden_dim, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, node_features, edge_index, edge_attr, post_mask):
        # Encode nodes
        node_emb = torch.relu(self.node_encoder(node_features))
        edge_emb = torch.relu(self.edge_encoder(edge_attr))
        
        # Message passing
        messages = torch.zeros_like(node_emb)
        src, dst = edge_index
        
        for i in range(edge_index.shape[1]):
            source = src[i]
            target = dst[i]
            message = torch.cat([node_emb[source], edge_emb[i]], dim=0)
            message = self.message_layer(message.unsqueeze(0)).squeeze(0)
            messages[target] += message
        
        # Update node embeddings
        node_emb = self.update_layer(messages, node_emb)
        
        # Predict on post nodes
        post_emb = node_emb[post_mask]
        out = torch.sigmoid(self.output_layer(post_emb).squeeze(-1))
        
        return out

# Simplified GCN model
class SimpleGCN(torch.nn.Module):
    def __init__(self, node_dim, hidden_dim=32):
        super(SimpleGCN, self).__init__()
        self.encoder = torch.nn.Linear(node_dim, hidden_dim)
        self.conv = torch.nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, node_features, edge_index, post_mask):
        # Encode nodes
        node_emb = torch.relu(self.encoder(node_features))
        
        # Message passing
        src, dst = edge_index
        messages = torch.zeros_like(node_emb)
        
        for i in range(edge_index.shape[1]):
            messages[dst[i]] += node_emb[src[i]]
        
        node_emb = torch.relu(self.conv(node_emb + messages))
        
        # Predict on post nodes
        post_emb = node_emb[post_mask]
        out = torch.sigmoid(self.output_layer(post_emb).squeeze(-1))
        
        return out

def create_data_subset(graph_data, subset_size=0.1):
    """Create a subset of the graph data for faster experiments."""
    print(f"Creating a subset with {subset_size*100:.1f}% of the data...")
    
    # Create a copy of essential data
    subset = {}
    
    # Keep node features, types, and labels
    subset['node_features'] = graph_data['node_features']
    subset['node_types'] = graph_data['node_types']
    subset['labels'] = graph_data['labels']
    subset['post_mask'] = graph_data['post_mask']
    subset['num_nodes'] = graph_data['num_nodes']
    
    # Sample a subset of edges
    n_edges = graph_data['edge_index'].shape[1]
    sample_size = int(n_edges * subset_size)
    edge_indices = np.random.choice(n_edges, sample_size, replace=False)
    
    # Keep the selected edges and related data
    subset['edge_index'] = graph_data['edge_index'][:, edge_indices]
    subset['edge_attr'] = graph_data['edge_attr'][edge_indices]
    subset['edge_times'] = graph_data['edge_times'][edge_indices]
    subset['edge_types'] = graph_data['edge_types'][edge_indices]
    
    print(f"Created subset with {sample_size} edges out of {n_edges} total edges")
    
    return subset

def compare_models(graph_data, epochs=20):
    """Train both models and compare their performance."""
    # Create train/val/test splits
    edge_times = graph_data['edge_times'].numpy()
    sorted_indices = np.argsort(edge_times)
    
    n_edges = len(edge_times)
    train_idx = int(n_edges * 0.7)
    val_idx = int(n_edges * 0.85)
    
    train_mask = sorted_indices[:train_idx]
    val_mask = sorted_indices[train_idx:val_idx]
    test_mask = sorted_indices[val_idx:]
    
    print(f"Splits: {len(train_mask)} train, {len(val_mask)} val, {len(test_mask)} test edges")
    
    # Get dimensions
    node_dim = graph_data['node_features'].shape[1]
    edge_dim = graph_data['edge_attr'].shape[1]
    
    # Load data to device
    node_features = graph_data['node_features'].to(device)
    labels = graph_data['labels'][graph_data['post_mask']].to(device)
    post_mask = graph_data['post_mask'].to(device)
    
    # Create edges for each split
    train_edge_index = graph_data['edge_index'][:, train_mask].to(device)
    train_edge_attr = graph_data['edge_attr'][train_mask].to(device)
    
    val_edge_index = graph_data['edge_index'][:, val_mask].to(device)
    val_edge_attr = graph_data['edge_attr'][val_mask].to(device)
    
    test_edge_index = graph_data['edge_index'][:, test_mask].to(device)
    test_edge_attr = graph_data['edge_attr'][test_mask].to(device)
    
    # Initialize models
    tgn = SimpleTGN(node_dim, edge_dim).to(device)
    gcn = SimpleGCN(node_dim).to(device)
    
    # Initialize optimizers
    tgn_optimizer = torch.optim.Adam(tgn.parameters(), lr=0.01)
    gcn_optimizer = torch.optim.Adam(gcn.parameters(), lr=0.01)
    
    # Loss function
    loss_fn = torch.nn.BCELoss()
    
    # Training loop
    tgn_train_losses = []
    tgn_val_losses = []
    tgn_test_metrics = {'accuracy': [], 'f1': [], 'auc': []}
    
    gcn_train_losses = []
    gcn_val_losses = []
    gcn_test_metrics = {'accuracy': [], 'f1': [], 'auc': []}
    
    print("\nTraining models...")
    for epoch in tqdm(range(epochs)):
        # Train TGN
        tgn.train()
        tgn_optimizer.zero_grad()
        tgn_outputs = tgn(node_features, train_edge_index, train_edge_attr, post_mask)
        tgn_loss = loss_fn(tgn_outputs, labels.float())
        tgn_loss.backward()
        tgn_optimizer.step()
        
        tgn_train_losses.append(tgn_loss.item())
        
        # Train GCN
        gcn.train()
        gcn_optimizer.zero_grad()
        gcn_outputs = gcn(node_features, train_edge_index, post_mask)
        gcn_loss = loss_fn(gcn_outputs, labels.float())
        gcn_loss.backward()
        gcn_optimizer.step()
        
        gcn_train_losses.append(gcn_loss.item())
        
        # Evaluate
        tgn.eval()
        gcn.eval()
        
        with torch.no_grad():
            # Validation losses
            tgn_val_outputs = tgn(node_features, val_edge_index, val_edge_attr, post_mask)
            tgn_val_loss = loss_fn(tgn_val_outputs, labels.float())
            tgn_val_losses.append(tgn_val_loss.item())
            
            gcn_val_outputs = gcn(node_features, val_edge_index, post_mask)
            gcn_val_loss = loss_fn(gcn_val_outputs, labels.float())
            gcn_val_losses.append(gcn_val_loss.item())
            
            # Test metrics
            tgn_test_outputs = tgn(node_features, test_edge_index, test_edge_attr, post_mask)
            gcn_test_outputs = gcn(node_features, test_edge_index, post_mask)
            
            tgn_preds = (tgn_test_outputs > 0.5).float().cpu().numpy()
            gcn_preds = (gcn_test_outputs > 0.5).float().cpu().numpy()
            
            test_labels = labels.cpu().numpy()
            
            # Calculate metrics
            tgn_test_metrics['accuracy'].append(accuracy_score(test_labels, tgn_preds))
            tgn_test_metrics['f1'].append(f1_score(test_labels, tgn_preds, zero_division=0))
            tgn_test_metrics['auc'].append(roc_auc_score(test_labels, tgn_test_outputs.cpu().numpy()))
            
            gcn_test_metrics['accuracy'].append(accuracy_score(test_labels, gcn_preds))
            gcn_test_metrics['f1'].append(f1_score(test_labels, gcn_preds, zero_division=0))
            gcn_test_metrics['auc'].append(roc_auc_score(test_labels, gcn_test_outputs.cpu().numpy()))
    
    # Final evaluation
    tgn.eval()
    gcn.eval()
    
    with torch.no_grad():
        tgn_test_outputs = tgn(node_features, test_edge_index, test_edge_attr, post_mask)
        gcn_test_outputs = gcn(node_features, test_edge_index, post_mask)
        
        tgn_preds = (tgn_test_outputs > 0.5).float().cpu().numpy()
        gcn_preds = (gcn_test_outputs > 0.5).float().cpu().numpy()
        
        test_labels = labels.cpu().numpy()
        
        tgn_accuracy = accuracy_score(test_labels, tgn_preds)
        tgn_f1 = f1_score(test_labels, tgn_preds, zero_division=0)
        tgn_auc = roc_auc_score(test_labels, tgn_test_outputs.cpu().numpy())
        
        gcn_accuracy = accuracy_score(test_labels, gcn_preds)
        gcn_f1 = f1_score(test_labels, gcn_preds, zero_division=0)
        gcn_auc = roc_auc_score(test_labels, gcn_test_outputs.cpu().numpy())
    
    # Print results
    print("\n=== Final Results ===")
    print("TGN:")
    print(f"  Accuracy: {tgn_accuracy:.4f}")
    print(f"  F1 Score: {tgn_f1:.4f}")
    print(f"  AUC: {tgn_auc:.4f}")
    
    print("GCN:")
    print(f"  Accuracy: {gcn_accuracy:.4f}")
    print(f"  F1 Score: {gcn_f1:.4f}")
    print(f"  AUC: {gcn_auc:.4f}")
    
    # Plot results
    os.makedirs('results', exist_ok=True)
    
    # Training and validation loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(tgn_train_losses, label='TGN Train')
    plt.plot(tgn_val_losses, label='TGN Val')
    plt.plot(gcn_train_losses, label='GCN Train')
    plt.plot(gcn_val_losses, label='GCN Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(tgn_test_metrics['auc'], label='TGN AUC')
    plt.plot(gcn_test_metrics['auc'], label='GCN AUC')
    plt.plot(tgn_test_metrics['f1'], label='TGN F1')
    plt.plot(gcn_test_metrics['f1'], label='GCN F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Test Metrics')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/quick_experiment_metrics.png')
    
    # Bar chart comparison
    plt.figure(figsize=(10, 6))
    metrics = ['Accuracy', 'F1', 'AUC']
    tgn_scores = [tgn_accuracy, tgn_f1, tgn_auc]
    gcn_scores = [gcn_accuracy, gcn_f1, gcn_auc]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, tgn_scores, width, label='TGN')
    plt.bar(x + width/2, gcn_scores, width, label='GCN')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('TGN vs GCN Performance')
    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/quick_experiment_comparison.png')
    
    print("\nResults saved to results directory.")
    
    return {
        'tgn': {
            'accuracy': tgn_accuracy, 
            'f1': tgn_f1, 
            'auc': tgn_auc
        },
        'gcn': {
            'accuracy': gcn_accuracy, 
            'f1': gcn_f1, 
            'auc': gcn_auc
        }
    }

def main():
    """Run quick experiment."""
    print("Starting quick experiment...")
    
    # Load the processed FakeNewsNet data
    data_path = 'data/processed/fakenewsnet_processed.pkl'
    if not os.path.exists(data_path):
        print(f"Error: Processed data not found at {data_path}")
        return
    
    with open(data_path, 'rb') as f:
        graph_data = pickle.load(f)
    
    print(f"Loaded graph with {graph_data['num_nodes']} nodes and {graph_data['edge_index'].shape[1]} edges")
    
    # Create a subset for faster experimentation
    subset_data = create_data_subset(graph_data, subset_size=0.05)
    
    # Compare models
    results = compare_models(subset_data, epochs=10)
    
    print("\nExperiment completed!")

if __name__ == "__main__":
    main() 