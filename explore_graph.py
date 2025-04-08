import pickle
import os
import numpy as np
import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the processed data
data_path = 'data/processed/coaid_processed.pkl'
with open(data_path, 'rb') as f:
    graph_data = pickle.load(f)

# Display basic information
print("===== CoAID Graph Dataset Summary =====")
print(f"Number of nodes: {graph_data['num_nodes']}")
print(f"Number of edges: {graph_data['edge_index'].shape[1]}")
print(f"Node feature dimension: {graph_data['node_features'].shape[1]}")
print(f"Edge feature dimension: {graph_data['edge_attr'].shape[1]}")

# Count node types
user_nodes = (graph_data['node_types'] == 0).sum().item()
post_nodes = (graph_data['node_types'] == 1).sum().item()
print(f"User nodes: {user_nodes}")
print(f"Post nodes: {post_nodes}")

# Count edge types
edge_type_counts = {}
for i, edge_type in enumerate(graph_data['edge_types'].numpy()):
    if edge_type not in edge_type_counts:
        edge_type_counts[edge_type] = 0
    edge_type_counts[edge_type] += 1

print("\nEdge type distribution:")
print("Type 0 (Like): ", edge_type_counts.get(0, 0))
print("Type 1 (Share): ", edge_type_counts.get(1, 0))
print("Type 2 (Comment): ", edge_type_counts.get(2, 0))
print("Type 3 (Other): ", edge_type_counts.get(3, 0))
print("Type 4 (Authorship): ", edge_type_counts.get(4, 0))

# Check class balance
labels = graph_data['labels']
post_mask = graph_data['post_mask']
post_labels = labels[post_mask]
misinformation_count = (post_labels == 1).sum().item()
reliable_count = (post_labels == 0).sum().item()

print("\nClass distribution (posts only):")
print(f"Misinformation: {misinformation_count} ({misinformation_count/len(post_labels)*100:.2f}%)")
print(f"Reliable: {reliable_count} ({reliable_count/len(post_labels)*100:.2f}%)")

# Explore temporal aspects
edge_times = graph_data['edge_times'].numpy()
print("\nTemporal information:")
print(f"Time range: {pd.to_datetime(min(edge_times), unit='s')} to {pd.to_datetime(max(edge_times), unit='s')}")
print(f"Duration: {max(edge_times) - min(edge_times):.2f} seconds ({(max(edge_times) - min(edge_times))/86400:.2f} days)")

# Create a NetworkX graph for visualization
G = nx.Graph()

# Add nodes
for i in range(graph_data['num_nodes']):
    node_type = "User" if graph_data['node_types'][i] == 0 else "Post"
    label = graph_data['labels'][i].item() if graph_data['post_mask'][i] else None
    G.add_node(i, type=node_type, label=label)

# Add edges
edge_index = graph_data['edge_index'].t().numpy()
for i in range(edge_index.shape[0]):
    src, dst = edge_index[i]
    edge_type = graph_data['edge_types'][i].item()
    G.add_edge(src, dst, type=edge_type)

# Plot the graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)

# Color nodes by type and label
node_colors = []
for n in G.nodes():
    if G.nodes[n]['type'] == "User":
        node_colors.append('skyblue')
    else:  # Post
        if G.nodes[n]['label'] == 1:
            node_colors.append('red')  # Misinformation
        else:
            node_colors.append('green')  # Reliable

nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=50, alpha=0.8)

# Add a legend
user_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=10, label='User')
reliable_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Reliable Post')
misinfo_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Misinfo Post')
plt.legend(handles=[user_patch, reliable_patch, misinfo_patch], loc='upper right')

plt.title('CoAID Network Graph Visualization')
plt.savefig('coaid_graph.png', dpi=300)
print("\nGraph visualization saved to 'coaid_graph.png'")

# Print summary of snapshots
snapshots = graph_data.get('snapshots', [])
print(f"\nNumber of temporal snapshots: {len(snapshots)}")
for i, snapshot in enumerate(snapshots):
    start_time = pd.to_datetime(snapshot['time_window'][0], unit='s')
    end_time = pd.to_datetime(snapshot['time_window'][1], unit='s')
    print(f"Snapshot {i+1}: {start_time} to {end_time}, Edges: {snapshot['edge_index'].shape[1]}")

print("\nExploration complete!") 