"""Test script for FakeNewsNet dataset processor."""

import os
from fakenewsnet_processor import FakeNewsNetProcessor

def test_dataset(dataset_type: str):
    """Test loading and processing for a specific dataset type."""
    print(f"\nTesting {dataset_type.upper()} dataset:")
    print("-" * 50)
    
    # Initialize processor with data directory
    data_dir = os.path.join('data', 'raw', 'fakenewsnet')
    processor = FakeNewsNetProcessor(data_dir, dataset_type=dataset_type)
    
    # Load all data
    print("Loading data...")
    processor.load_all_data()
    
    # Print basic statistics
    print("\nDataset Statistics:")
    print(f"Number of news articles: {len(processor.news_data) if processor.news_data is not None else 0}")
    print(f"Number of tweets: {len(processor.tweet_data) if processor.tweet_data is not None else 0}")
    print(f"Number of unique users: {len(processor.user_data) if processor.user_data is not None else 0}")
    
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

def main():
    # Test both PolitiFact and GossipCop datasets
    test_dataset("politifact")
    test_dataset("gossipcop")

if __name__ == "__main__":
    main() 