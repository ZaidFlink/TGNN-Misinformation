"""
CoAID dataset processor for temporal graph neural networks.
Handles loading and preprocessing of the CoAID COVID-19 misinformation dataset.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import torch
from torch_geometric.data import Data
import re

class CoAIDProcessor:
    """Processor for the CoAID dataset to create temporal graphs."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the CoAID dataset processor.
        
        Args:
            data_dir: Path to the root directory containing CoAID data
        """
        self.data_dir = data_dir
        self.time_periods = ['05-01-2020', '07-01-2020', '09-01-2020', '11-01-2020']
        
        # Storage for processed data
        self.news_data = None
        self.tweets_data = None
        self.replies_data = None
        
    def _clean_date_string(self, date_str: str) -> str:
        """Clean date string by removing timezone and extra information."""
        if pd.isna(date_str):
            return date_str
            
        # Remove timezone abbreviations (PDT, EDT, etc.)
        date_str = re.sub(r'\s+[A-Z]{3}(\s|$)', ' ', date_str)
        
        # Remove content in square brackets
        date_str = re.sub(r'\s*\[.*?\]\s*', '', date_str)
        
        # Remove "at" from date strings
        date_str = date_str.replace(' at ', ' ')
        
        # Handle short month format without year
        if re.match(r'^\d{1,2}-[A-Za-z]{3}$', date_str):
            date_str = f"{date_str}-2020"
            
        return date_str.strip()
    
    def _parse_date(self, date_str: str) -> pd.Timestamp:
        """
        Parse date strings in various formats to pandas Timestamp.
        
        Args:
            date_str: Date string to parse
            
        Returns:
            pd.Timestamp object
        """
        if pd.isna(date_str):
            return pd.NaT
            
        # Clean the date string
        date_str = self._clean_date_string(date_str)
        
        try:
            # Try common formats
            formats = [
                "%B %d, %Y %I:%M %p",     # "April 17, 2020 5:47 pm"
                "%Y-%m-%d",               # "2020-04-17"
                "%Y-%m-%dT%H:%M:%S",      # ISO format without timezone
                "%Y-%m-%d %H:%M:%S",      # Standard datetime
                "%d-%b-%Y",               # "06-Mar-2020"
                "%B %d, %Y",              # "March 6, 2020"
                "%b %d, %Y %I:%M%p",      # "Mar 6, 2020 2:18PM"
                "%b %d, %Y",              # "Mar 6, 2020"
                "%d-%b-%Y"                # "15-May-2020"
            ]
            
            for fmt in formats:
                try:
                    dt = pd.to_datetime(date_str, format=fmt)
                    # Convert to UTC and make naive
                    if dt.tz is not None:
                        dt = dt.tz_convert('UTC').tz_localize(None)
                    return dt
                except ValueError:
                    continue
                    
            # If none of the specific formats work, try the flexible parser
            # with error handling for relative dates
            if any(x in date_str.lower() for x in ['ago', 'month', 'day', 'hour', 'min']):
                return pd.NaT
                
            dt = pd.to_datetime(date_str)
            if dt.tz is not None:
                dt = dt.tz_convert('UTC').tz_localize(None)
            return dt
            
        except Exception as e:
            print(f"Warning: Could not parse date '{date_str}': {str(e)}")
            return pd.NaT
    
    def load_period_data(self, period: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load data for a specific time period.
        
        Args:
            period: Time period directory name (e.g., '05-01-2020')
            
        Returns:
            Tuple of DataFrames (news, tweets, replies)
        """
        period_dir = os.path.join(self.data_dir, period)
        
        # Load and combine fake and real news
        fake_news = pd.read_csv(os.path.join(period_dir, 'NewsFakeCOVID-19.csv'))
        real_news = pd.read_csv(os.path.join(period_dir, 'NewsRealCOVID-19.csv'))
        fake_news['label'] = 1  # Fake
        real_news['label'] = 0  # Real
        news = pd.concat([fake_news, real_news], ignore_index=True)
        
        # Load tweets
        fake_tweets = pd.read_csv(os.path.join(period_dir, 'NewsFakeCOVID-19_tweets.csv'))
        real_tweets = pd.read_csv(os.path.join(period_dir, 'NewsRealCOVID-19_tweets.csv'))
        tweets = pd.concat([fake_tweets, real_tweets], ignore_index=True)
        
        # Load replies
        fake_replies = pd.read_csv(os.path.join(period_dir, 'NewsFakeCOVID-19_tweets_replies.csv'))
        real_replies = pd.read_csv(os.path.join(period_dir, 'NewsRealCOVID-19_tweets_replies.csv'))
        replies = pd.concat([fake_replies, real_replies], ignore_index=True)
        
        # Add period information
        news['period'] = period
        tweets['period'] = period
        replies['period'] = period
        
        return news, tweets, replies
    
    def load_all_data(self):
        """Load and combine data from all time periods."""
        all_news = []
        all_tweets = []
        all_replies = []
        
        for period in self.time_periods:
            news, tweets, replies = self.load_period_data(period)
            all_news.append(news)
            all_tweets.append(tweets)
            all_replies.append(replies)
        
        self.news_data = pd.concat(all_news, ignore_index=True)
        self.tweets_data = pd.concat(all_tweets, ignore_index=True)
        self.replies_data = pd.concat(all_replies, ignore_index=True)
        
        # Parse dates and convert to datetime
        self.news_data['publish_date'] = self.news_data['publish_date'].apply(self._parse_date)
        
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
            # This is a placeholder - you'll want to implement more sophisticated edge creation
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