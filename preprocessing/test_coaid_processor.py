"""Test script for CoAID dataset processor."""

import os
from coaid_processor import CoAIDProcessor

def main():
    # Initialize processor with data directory
    data_dir = os.path.join('data', 'raw', 'coaid')
    processor = CoAIDProcessor(data_dir)
    
    # Load all data
    print("Loading data...")
    processor.load_all_data()
    
    # Print basic statistics
    print("\nDataset Statistics:")
    print(f"Number of news articles: {len(processor.news_data)}")
    print(f"Number of tweets: {len(processor.tweets_data)}")
    print(f"Number of replies: {len(processor.replies_data)}")
    
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
        print(f"  Labels shape: {graph.y.shape}")
        print(f"  Timestamp: {graph.timestamp}")

if __name__ == "__main__":
    main() 