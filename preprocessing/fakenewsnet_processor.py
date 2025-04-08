"""
FakeNewsNet dataset processor for temporal graph neural networks.
Handles loading and preprocessing of the FakeNewsNet dataset (PolitiFact and GossipCop).
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import torch
from torch_geometric.data import Data
import json

class FakeNewsNetProcessor:
    """Processor for the FakeNewsNet dataset to create temporal graphs."""
    
    def __init__(self, data_dir: str, dataset_type: str = "politifact"):
        """
        Initialize the FakeNewsNet dataset processor.
        
        Args:
            data_dir: Path to the root directory containing FakeNewsNet data
            dataset_type: Which dataset to use - "politifact" or "gossipcop"
        """
        if dataset_type not in ["politifact", "gossipcop"]:
            raise ValueError("dataset_type must be either 'politifact' or 'gossipcop'")
            
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        
        # Storage for processed data
        self.news_data = None
        self.tweet_data = None
        self.user_data = None
        
    def load_news_content(self, label: str, news_id: str) -> Dict:
        """
        Load news content for a specific news article.
        
        Args:
            label: "fake" or "real"
            news_id: ID of the news article
            
        Returns:
            Dict containing news content or empty dict if file doesn't exist
        """
        content_path = os.path.join(
            self.data_dir,
            self.dataset_type,
            label,
            str(news_id),
            "news content.json"
        )
        
        try:
            with open(content_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
            
    def load_tweets(self, label: str, news_id: str) -> List[Dict]:
        """
        Load all tweets for a specific news article.
        
        Args:
            label: "fake" or "real"
            news_id: ID of the news article
            
        Returns:
            List of tweet objects
        """
        tweets_dir = os.path.join(
            self.data_dir,
            self.dataset_type,
            label,
            str(news_id),
            "tweets"
        )
        
        tweets = []
        if os.path.exists(tweets_dir):
            for tweet_file in os.listdir(tweets_dir):
                if tweet_file.endswith('.json'):
                    try:
                        with open(os.path.join(tweets_dir, tweet_file), 'r', encoding='utf-8') as f:
                            tweets.append(json.load(f))
                    except (json.JSONDecodeError, FileNotFoundError):
                        continue
                        
        return tweets
        
    def load_all_data(self):
        """Load and combine all news and social context data."""
        news_list = []
        tweet_list = []
        user_set = set()
        
        # Load the CSV files containing news information
        fake_df = pd.read_csv(os.path.join(self.data_dir, f"{self.dataset_type}_fake.csv"))
        real_df = pd.read_csv(os.path.join(self.data_dir, f"{self.dataset_type}_real.csv"))
        
        # Add labels
        fake_df['label'] = 1
        real_df['label'] = 0
        
        # Process fake news
        for _, row in fake_df.iterrows():
            news_content = self.load_news_content("fake", row['id'])
            if news_content:
                news_content.update({
                    'id': row['id'],
                    'url': row['url'],
                    'title': row['title'],
                    'label': 1
                })
                news_list.append(news_content)
                
                # Load associated tweets
                tweets = self.load_tweets("fake", row['id'])
                for tweet in tweets:
                    tweet['news_id'] = row['id']
                    tweet_list.append(tweet)
                    if 'user' in tweet:
                        user_set.add(tweet['user']['id_str'])
                        
        # Process real news
        for _, row in real_df.iterrows():
            news_content = self.load_news_content("real", row['id'])
            if news_content:
                news_content.update({
                    'id': row['id'],
                    'url': row['url'],
                    'title': row['title'],
                    'label': 0
                })
                news_list.append(news_content)
                
                # Load associated tweets
                tweets = self.load_tweets("real", row['id'])
                for tweet in tweets:
                    tweet['news_id'] = row['id']
                    tweet_list.append(tweet)
                    if 'user' in tweet:
                        user_set.add(tweet['user']['id_str'])
                        
        # Convert to DataFrames
        self.news_data = pd.DataFrame(news_list)
        self.tweet_data = pd.DataFrame(tweet_list)
        
        # Convert publish dates to datetime
        if 'publish date' in self.news_data.columns:
            self.news_data['publish_date'] = pd.to_datetime(
                self.news_data['publish date'],
                utc=True
            ).dt.tz_localize(None)  # Convert to naive datetime
            
        # Store unique users
        self.user_data = pd.DataFrame({'user_id': list(user_set)})
        
    def create_temporal_graph(self, time_window: str = 'D') -> List[Data]:
        """
        Create a list of temporal graphs based on the specified time window.
        
        Args:
            time_window: Pandas time frequency string ('D' for daily, 'W' for weekly)
            
        Returns:
            List of PyTorch Geometric Data objects representing temporal graphs
        """
        if self.news_data is None:
            self.load_all_data()
            
        # Drop rows with invalid dates
        valid_dates = self.news_data.dropna(subset=['publish_date'])
        
        # Group data by time window
        grouped = valid_dates.groupby(pd.Grouper(key='publish_date', freq=time_window))
        
        temporal_graphs = []
        for _, period_data in grouped:
            if len(period_data) == 0:
                continue
                
            # Create node features
            # For now, using simple one-hot encoding for demonstration
            num_nodes = len(period_data)
            node_features = torch.eye(num_nodes)  # One-hot features
            
            # Create edges based on content similarity
            edge_index = self._create_edges(period_data)
            
            # Create labels
            labels = torch.tensor(period_data['label'].values, dtype=torch.float)
            
            # Create PyG Data object
            graph = Data(
                x=node_features,
                edge_index=edge_index,
                y=labels,
                timestamp=period_data['publish_date'].min().timestamp()
            )
            
            temporal_graphs.append(graph)
            
        return temporal_graphs
        
    def _create_edges(self, period_data: pd.DataFrame) -> torch.Tensor:
        """
        Create edges between news articles based on content similarity.
        This is a placeholder implementation - you should implement more 
        sophisticated edge creation based on your needs.
        
        Args:
            period_data: DataFrame containing news articles for a time period
            
        Returns:
            torch.Tensor: Edge index tensor of shape [2, num_edges]
        """
        # Placeholder: Create a simple chain of edges
        num_nodes = len(period_data)
        if num_nodes < 2:
            return torch.zeros((2, 0), dtype=torch.long)
            
        source = torch.arange(num_nodes - 1)
        target = torch.arange(1, num_nodes)
        edge_index = torch.stack([source, target])
        
        return edge_index 