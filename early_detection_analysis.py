#!/usr/bin/env python
"""
Early Detection Analysis Script

This script analyzes the early detection capabilities of the TGN and GCN models
by evaluating how quickly they can identify misinformation.
"""

import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from train_small_dataset import SimplifiedTGN, SimplifiedGCN, load_data

def time_to_detection(model, graph_data, model_type='tgn'):
    """Analyze how quickly a model can detect misinformation after the first interaction."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Load data
    node_features = graph_data['node_features'].to(device)
    edge_index = graph_data['edge_index'].to(device)
    edge_attr = graph_data['edge_attr'].to(device)
    edge_times = graph_data['edge_times'].cpu().numpy()
    post_mask = graph_data['post_mask'].to(device)
    labels = graph_data['labels'][post_mask].cpu().numpy()
    
    # Get post IDs
    post_ids = []
    for i, is_post in enumerate(post_mask.cpu().numpy()):
        if is_post:
            post_ids.append(i)
    
    # Get mapping from node IDs to original IDs
    id_to_node = graph_data.get('id_to_node', {})
    
    # Dictionary to track first interaction time for each post
    first_interaction_times = {}
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i].cpu().numpy()
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
    time_windows = []
    min_time = min(edge_times)
    max_time = max(edge_times)
    total_duration = max_time - min_time
    
    # Create 10 time windows
    for i in range(10):
        window_start = min_time + i * (total_duration / 10)
        window_end = min_time + (i + 1) * (total_duration / 10)
        time_windows.append((window_start, window_end))
    
    # Results for each time window
    detection_results = []
    
    # For each time window, evaluate the model's detection capability
    for window_idx, (start_time, end_time) in enumerate(time_windows):
        # Get edges within this time window
        time_mask = (edge_times >= min_time) & (edge_times <= end_time)
        window_edge_index = edge_index[:, time_mask].to(device)
        window_edge_attr = edge_attr[time_mask].to(device)
        
        # Skip if no edges
        if window_edge_index.shape[1] == 0:
            continue
        
        # Predict
        with torch.no_grad():
            if model_type == 'tgn':
                predictions = model(node_features, window_edge_index, window_edge_attr, post_mask)
            else:  # GCN
                predictions = model(node_features, window_edge_index, post_mask)
        
        # Calculate metrics for misinformation posts
        predictions = predictions.cpu().numpy()
        pred_labels = (predictions > 0.5).astype(int)
        
        # For each post, check if correctly identified as misinformation
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
                    'post_id': id_to_node.get(post_id, post_id),
                    'window_idx': window_idx,
                    'time_window': f"{pd.to_datetime(start_time, unit='s')} - {pd.to_datetime(end_time, unit='s')}",
                    'time_since_first_interaction': time_since_first,
                    'time_since_first_hours': time_since_first / 3600,
                    'correctly_detected': correctly_detected,
                    'confidence': predictions[i]
                })
    
    return pd.DataFrame(detection_results)

def visualize_early_detection(tgn_results, gcn_results):
    """Visualize early detection results comparing TGN and GCN."""
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot 1: Detection rate over time
    plt.figure(figsize=(12, 6))
    
    # Group by time window and calculate detection rate
    tgn_by_window = tgn_results.groupby('window_idx')['correctly_detected'].mean()
    gcn_by_window = gcn_results.groupby('window_idx')['correctly_detected'].mean()
    
    plt.plot(tgn_by_window.index, tgn_by_window.values, 'o-', label='TGN', linewidth=2)
    plt.plot(gcn_by_window.index, gcn_by_window.values, 's-', label='GCN', linewidth=2)
    
    plt.xlabel('Time Window')
    plt.ylabel('Detection Rate')
    plt.title('Misinformation Detection Rate Over Time')
    plt.xticks(tgn_by_window.index)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig('early_detection_rate.png')
    
    # Plot 2: Confidence over time
    plt.figure(figsize=(12, 6))
    
    # Group by time window and calculate average confidence
    tgn_conf_by_window = tgn_results.groupby('window_idx')['confidence'].mean()
    gcn_conf_by_window = gcn_results.groupby('window_idx')['confidence'].mean()
    
    plt.plot(tgn_conf_by_window.index, tgn_conf_by_window.values, 'o-', label='TGN', linewidth=2)
    plt.plot(gcn_conf_by_window.index, gcn_conf_by_window.values, 's-', label='GCN', linewidth=2)
    
    plt.xlabel('Time Window')
    plt.ylabel('Confidence')
    plt.title('Model Confidence in Misinformation Detection Over Time')
    plt.xticks(tgn_conf_by_window.index)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig('early_detection_confidence.png')
    
    # Plot 3: Time to correct detection
    plt.figure(figsize=(12, 6))
    
    # Filter for correctly detected posts
    tgn_correct = tgn_results[tgn_results['correctly_detected']]
    gcn_correct = gcn_results[gcn_results['correctly_detected']]
    
    # Group by post and get first correct detection
    tgn_first_correct = tgn_correct.groupby('post_id')['time_since_first_hours'].min()
    gcn_first_correct = gcn_correct.groupby('post_id')['time_since_first_hours'].min()
    
    # Calculate average time to detection
    tgn_avg_time = tgn_first_correct.mean()
    gcn_avg_time = gcn_first_correct.mean()
    
    bars = plt.bar(['TGN', 'GCN'], [tgn_avg_time, gcn_avg_time], color=['#1f77b4', '#ff7f0e'])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f} hours', ha='center', va='bottom')
    
    plt.ylabel('Average Time to Detection (hours)')
    plt.title('Average Time to Correct Misinformation Detection')
    plt.ylim(0, max(tgn_avg_time, gcn_avg_time) * 1.2)
    plt.tight_layout()
    plt.savefig('time_to_detection.png')
    
    return {
        'tgn_avg_detection_time': tgn_avg_time,
        'gcn_avg_detection_time': gcn_avg_time,
        'tgn_final_detection_rate': tgn_by_window.iloc[-1] if len(tgn_by_window) > 0 else 0,
        'gcn_final_detection_rate': gcn_by_window.iloc[-1] if len(gcn_by_window) > 0 else 0
    }

def main():
    # Load data
    print("Loading CoAID processed data...")
    graph_data = load_data()
    
    # Get dimensions
    node_dim = graph_data['node_features'].shape[1]
    edge_dim = graph_data['edge_attr'].shape[1]
    
    # Create and train TGN model (reuse the trained model if you saved it)
    print("Initializing TGN model...")
    tgn_model = SimplifiedTGN(node_dim, edge_dim)
    
    # Create and train GCN model
    print("Initializing GCN model...")
    gcn_model = SimplifiedGCN(node_dim)
    
    # Analyze early detection capabilities
    print("Analyzing TGN early detection capability...")
    tgn_results = time_to_detection(tgn_model, graph_data, model_type='tgn')
    
    print("Analyzing GCN early detection capability...")
    gcn_results = time_to_detection(gcn_model, graph_data, model_type='gcn')
    
    # Display results
    print("\nTGN Early Detection Results:")
    print(tgn_results)
    
    print("\nGCN Early Detection Results:")
    print(gcn_results)
    
    # Visualize results
    print("\nCreating visualizations...")
    metrics = visualize_early_detection(tgn_results, gcn_results)
    
    print("\n=== Early Detection Comparison ===")
    print(f"TGN avg time to detection: {metrics['tgn_avg_detection_time']:.2f} hours")
    print(f"GCN avg time to detection: {metrics['gcn_avg_detection_time']:.2f} hours")
    print(f"TGN final detection rate: {metrics['tgn_final_detection_rate']:.2%}")
    print(f"GCN final detection rate: {metrics['gcn_final_detection_rate']:.2%}")
    
    if metrics['tgn_avg_detection_time'] < metrics['gcn_avg_detection_time']:
        time_diff = metrics['gcn_avg_detection_time'] - metrics['tgn_avg_detection_time']
        print(f"\nTGN detected misinformation {time_diff:.2f} hours earlier than GCN on average.")
    elif metrics['gcn_avg_detection_time'] < metrics['tgn_avg_detection_time']:
        time_diff = metrics['tgn_avg_detection_time'] - metrics['gcn_avg_detection_time']
        print(f"\nGCN detected misinformation {time_diff:.2f} hours earlier than TGN on average.")
    else:
        print("\nBoth models detected misinformation at the same time on average.")
    
    print("\nEarly detection analysis complete. Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main() 