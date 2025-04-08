"""
TGB dataset processor for temporal graph neural networks.
Handles loading and preprocessing of the Temporal Graph Benchmark dataset.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import torch
from torch_geometric.data import Data

class TGBProcessor:
    """Processor for the TGB dataset to create temporal graphs."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the TGB dataset processor.
        
        Args:
            data_dir: Path to the root directory containing TGB data
        """
        self.data_dir = data_dir
        
        # Storage for processed data
        self.posts_data = None
        self.interactions_data = None
        
        # Mappings for categorical variables
        self.interaction_type_map = {
            'like': 0,
            'share': 1,
            'comment': 2
        }
        
    def load_all_data(self):
        """Load and combine all data."""
        # Load posts data
        posts_path = os.path.join(self.data_dir, 'posts.csv')
        self.posts_data = pd.read_csv(posts_path)
        
        # Load interactions data
        interactions_path = os.path.join(self.data_dir, 'interactions.csv')
        self.interactions_data = pd.read_csv(interactions_path)
        
        # Convert timestamps to datetime
        self.posts_data['timestamp'] = pd.to_datetime(self.posts_data['timestamp'])
        self.interactions_data['timestamp'] = pd.to_datetime(self.interactions_data['timestamp'])
        
        # Sort by timestamp
        self.posts_data = self.posts_data.sort_values('timestamp')
        self.interactions_data = self.interactions_data.sort_values('timestamp')
        
        # Convert interaction types to numeric
        if 'interaction_type' in self.interactions_data.columns:
            self.interactions_data['interaction_type'] = self.interactions_data['interaction_type'].map(
                self.interaction_type_map
            )
        
    def create_temporal_graph(self, time_window: str = 'D') -> List[Data]:
        """
        Create a list of temporal graphs based on the specified time window.
        
        Args:
            time_window: Pandas time frequency string ('D' for daily, 'W' for weekly)
            
        Returns:
            List of PyTorch Geometric Data objects representing temporal graphs
        """
        if self.interactions_data is None:
            self.load_all_data()
            
        # Group interactions by time window
        grouped = self.interactions_data.groupby(
            pd.Grouper(key='timestamp', freq=time_window)
        )
        
        temporal_graphs = []
        node_mapping = {}  # Map original node IDs to consecutive integers
        current_nodes = set()  # Track nodes seen so far
        
        # First, add all posts to the node mapping
        for post_id in self.posts_data['post_id'].unique():
            if post_id not in current_nodes:
                node_mapping[post_id] = len(node_mapping)
                current_nodes.add(post_id)
        
        for _, period_data in grouped:
            if len(period_data) == 0:
                continue
                
            # Get posts for this period
            period_start = period_data['timestamp'].min()
            period_end = period_data['timestamp'].max()
            period_posts = self.posts_data[
                (self.posts_data['timestamp'] >= period_start) &
                (self.posts_data['timestamp'] <= period_end)
            ]
            
            # Update node mapping with new users
            users = period_data['user_id'].unique()
            new_users = set(users) - current_nodes
            
            for user in new_users:
                node_mapping[user] = len(node_mapping)
                current_nodes.add(user)
                
            # Create edge index
            sources = torch.tensor([node_mapping[n] for n in period_data['user_id']], dtype=torch.long)
            targets = torch.tensor([node_mapping[n] for n in period_data['post_id']], dtype=torch.long)
            edge_index = torch.stack([sources, targets])
            
            # Create node features
            num_nodes = len(node_mapping)
            node_features = []
            
            for node_id, idx in node_mapping.items():
                # Check if node is a post
                post_row = self.posts_data[self.posts_data['post_id'] == node_id]
                if len(post_row) > 0:
                    # Post features: [is_post=1, is_user=0, is_misinformation]
                    features = [1, 0, int(post_row['is_misinformation'].iloc[0])]
                else:
                    # User features: [is_post=0, is_user=1, activity_count]
                    activity_count = len(period_data[period_data['user_id'] == node_id])
                    features = [0, 1, activity_count]
                node_features.append(features)
            
            node_features = torch.tensor(node_features, dtype=torch.float)
            
            # Create edge features if available
            if 'interaction_type' in period_data.columns:
                edge_attr = torch.tensor(period_data['interaction_type'].values, dtype=torch.float)
                edge_attr = edge_attr.view(-1, 1)  # Shape: [num_edges, 1]
            else:
                edge_attr = None
            
            # Create PyG Data object
            graph = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_nodes,
                timestamp=period_data['timestamp'].min().timestamp()
            )
            
            temporal_graphs.append(graph)
            
        return temporal_graphs
        
    def get_statistics(self) -> Dict[str, int]:
        """
        Get basic statistics about the dataset.
        
        Returns:
            Dict containing statistics
        """
        if self.posts_data is None or self.interactions_data is None:
            self.load_all_data()
            
        stats = {
            'num_posts': len(self.posts_data) if self.posts_data is not None else 0,
            'num_interactions': len(self.interactions_data) if self.interactions_data is not None else 0,
            'num_unique_users': len(self.interactions_data['user_id'].unique()) if self.interactions_data is not None else 0,
            'num_misinformation_posts': len(self.posts_data[self.posts_data['is_misinformation'] == 1]) if self.posts_data is not None else 0,
            'interaction_types': self.interactions_data['interaction_type'].value_counts().to_dict() if self.interactions_data is not None else {},
            'time_span': None
        }
        
        if self.interactions_data is not None and len(self.interactions_data) > 0:
            time_span = self.interactions_data['timestamp'].max() - self.interactions_data['timestamp'].min()
            stats['time_span'] = f"{time_span.total_seconds() / 3600:.1f} hours"
            
        return stats 