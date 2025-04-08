"""Test script for TGB dataset processor."""

import os
from tgb_processor import TGBProcessor

def main():
    """Test the TGB dataset processor."""
    print("Testing TGB dataset processor:")
    print("-" * 50)
    
    # Initialize processor
    data_dir = os.path.join('data', 'raw', 'tgb')
    processor = TGBProcessor(data_dir)
    
    # Get dataset statistics
    print("\nLoading data and computing statistics...")
    stats = processor.get_statistics()
    
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Create temporal graphs
    print("\nCreating temporal graphs...")
    temporal_graphs = processor.create_temporal_graph(time_window='W')  # Weekly graphs
    
    print(f"\nCreated {len(temporal_graphs)} temporal graphs")
    print("\nSample graph statistics:")
    for i, graph in enumerate(temporal_graphs[:3]):  # Show first 3 graphs
        print(f"\nGraph {i}:")
        print(f"  Nodes: {graph.num_nodes}")
        print(f"  Edges: {graph.num_edges}")
        print(f"  Features shape: {graph.x.shape}")
        if graph.edge_attr is not None:
            print(f"  Edge features shape: {graph.edge_attr.shape}")
        print(f"  Timestamp: {graph.timestamp}")

if __name__ == "__main__":
    main() 