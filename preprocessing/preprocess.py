"""
Preprocessing module for temporal graph-based misinformation detection.
This module handles loading datasets, constructing temporal graphs,
and engineering features for the models.
"""

import os
import pandas as pd
import numpy as np
import pickle
import yaml
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import torch
import networkx as nx
from tqdm import tqdm
import time
from datetime import datetime

# Download nltk resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


class DataPreprocessor:
    """Preprocessor for temporal graph data for misinformation detection."""
    
    def __init__(self, config_path='configs/config.yaml'):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config_path (str): Path to the configuration file.
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize text feature extraction models based on config
        self.init_text_feature_extractors()
        
    def init_text_feature_extractors(self):
        """Initialize text feature extraction models based on configuration."""
        text_config = self.config['data']['feature_engineering']['text']
        
        if text_config['use_tfidf']:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=text_config['max_features'],
                stop_words='english'
            )
        
        if text_config['use_pretrained']:
            model_name = text_config['pretrained_model']
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_model = AutoModel.from_pretrained(model_name)
    
    def load_dataset(self, dataset_name):
        """
        Load raw dataset files.
        
        Args:
            dataset_name (str): Name of the dataset ('coaid', 'fakenewsnet', or 'tgb').
            
        Returns:
            tuple: (posts_df, interactions_df) DataFrames containing posts and interactions.
        """
        base_path = os.path.join('data', 'raw', dataset_name)
        
        posts_path = os.path.join(base_path, 'posts.csv')
        interactions_path = os.path.join(base_path, 'interactions.csv')
        
        posts_df = pd.read_csv(posts_path)
        interactions_df = pd.read_csv(interactions_path)
        
        # Convert timestamps to datetime
        posts_df['timestamp'] = pd.to_datetime(posts_df['timestamp'])
        interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
        
        print(f"Loaded {len(posts_df)} posts and {len(interactions_df)} interactions from {dataset_name}")
        return posts_df, interactions_df
    
    def preprocess_text(self, text_series):
        """
        Preprocess text data for feature extraction.
        
        Args:
            text_series (pd.Series): Series containing text data.
            
        Returns:
            list: List of preprocessed text.
        """
        processed_texts = []
        
        for text in text_series:
            if not isinstance(text, str):
                processed_texts.append('')
                continue
                
            # Convert to lowercase and tokenize
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords and punctuation
            tokens = [word for word in tokens if word.isalnum() and word not in self.stop_words]
            
            # Rejoin tokens
            processed_text = ' '.join(tokens)
            processed_texts.append(processed_text)
            
        return processed_texts
    
    def extract_text_features(self, text_series):
        """
        Extract features from text using TF-IDF and/or pretrained models.
        
        Args:
            text_series (pd.Series): Series containing text data.
            
        Returns:
            np.ndarray: Matrix of text features.
        """
        text_config = self.config['data']['feature_engineering']['text']
        processed_texts = self.preprocess_text(text_series)
        
        features = []
        
        # Extract TF-IDF features if configured
        if text_config['use_tfidf']:
            tfidf_features = self.tfidf_vectorizer.fit_transform(processed_texts).toarray()
            features.append(tfidf_features)
        
        # Extract pretrained model features if configured
        if text_config['use_pretrained']:
            pretrained_features = []
            
            for text in tqdm(processed_texts, desc="Extracting text embeddings"):
                if not text:
                    # For empty text, use zero vector
                    pretrained_features.append(np.zeros(768))  # Common embedding dimension
                    continue
                    
                # Tokenize and get embeddings
                inputs = self.tokenizer(text, return_tensors="pt", 
                                        padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.text_model(**inputs)
                
                # Use CLS token embedding as the document representation
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                pretrained_features.append(embedding.flatten())
                
            pretrained_features = np.vstack(pretrained_features)
            features.append(pretrained_features)
        
        # Combine all features
        if len(features) > 1:
            combined_features = np.hstack(features)
        else:
            combined_features = features[0]
            
        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(combined_features)
        
        return normalized_features
    
    def encode_timestamps(self, timestamps):
        """
        Encode timestamps into high-dimensional vectors.
        
        Args:
            timestamps (pd.Series): Series of datetime timestamps.
            
        Returns:
            np.ndarray: Matrix of time encodings.
        """
        time_config = self.config['data']['feature_engineering']['time']
        encoding_dim = time_config['encoding_dim']
        
        # Convert to unix timestamps (seconds since epoch)
        unix_timestamps = timestamps.astype(np.int64) // 10**9
        
        # Initialize encoding matrix
        time_encodings = np.zeros((len(unix_timestamps), encoding_dim))
        
        # Use harmonic functions to encode time
        for i in range(encoding_dim // 2):
            omega = 1.0 / (10000 ** (2 * i / encoding_dim))
            time_encodings[:, 2*i] = np.sin(omega * unix_timestamps)
            time_encodings[:, 2*i+1] = np.cos(omega * unix_timestamps)
            
        return time_encodings
    
    def construct_temporal_graph(self, posts_df, interactions_df):
        """
        Construct a temporal graph from posts and interactions.
        
        Args:
            posts_df (pd.DataFrame): DataFrame containing posts.
            interactions_df (pd.DataFrame): DataFrame containing interactions.
            
        Returns:
            dict: A dictionary containing graph data:
                - node_features: Node feature matrix
                - edge_index: Edge connectivity
                - edge_attr: Edge features
                - timestamps: Edge timestamps
                - node_types: Type of each node (post or user)
                - edge_types: Type of each edge (like, share, comment)
                - labels: Labels for post nodes (1 for misinformation, 0 for reliable)
        """
        # Extract unique users and posts
        unique_users = set(interactions_df['user_id'].unique())
        unique_authors = set(posts_df['author_id'].unique())
        unique_posts = set(posts_df['post_id'].unique())
        
        # Combined set of all user nodes (including authors)
        all_users = unique_users.union(unique_authors)
        
        # Create node ID mappings
        user_to_id = {user: i for i, user in enumerate(all_users)}
        post_to_id = {post: i + len(user_to_id) for i, post in enumerate(unique_posts)}
        
        # Combine all nodes
        all_nodes = list(all_users) + list(unique_posts)
        num_nodes = len(all_nodes)
        
        # Create node type indicators (0 for users, 1 for posts)
        node_types = torch.zeros(num_nodes, dtype=torch.long)
        for i in range(len(user_to_id), num_nodes):
            node_types[i] = 1
        
        # Initialize node features
        node_features = np.zeros((num_nodes, self.config['models']['tgn']['node_dim']))
        
        # Extract text features for post nodes
        post_text_features = self.extract_text_features(posts_df['content'])
        
        # Assign post text features to post nodes
        for i, post_id in enumerate(posts_df['post_id']):
            if post_id in post_to_id:  # Check if post exists in mapping
                node_idx = post_to_id[post_id]
                feature_dim = min(post_text_features.shape[1], node_features.shape[1])
                node_features[node_idx, :feature_dim] = post_text_features[i, :feature_dim]
        
        # Create edges from interactions
        edge_index = []
        edge_attr = []
        edge_times = []
        edge_types = []
        
        # Add edges for each interaction
        for _, row in tqdm(interactions_df.iterrows(), desc="Processing interactions", 
                          total=len(interactions_df)):
            user_id = user_to_id[row['user_id']]
            post_id = post_to_id[row['post_id']]
            timestamp = row['timestamp'].timestamp()  # Convert to Unix timestamp
            
            # Map interaction type to integer
            if row['interaction_type'] == 'like':
                interaction_type = 0
            elif row['interaction_type'] == 'share':
                interaction_type = 1
            elif row['interaction_type'] == 'comment':
                interaction_type = 2
            else:
                interaction_type = 3  # Other
            
            # Add edges in both directions (user -> post and post -> user)
            edge_index.append((user_id, post_id))
            edge_index.append((post_id, user_id))
            
            # Add edge features and metadata for both directions
            time_encoded = self.encode_timestamps(pd.Series([row['timestamp']]))[0]
            edge_attr.append(np.concatenate([time_encoded, [interaction_type]]))
            edge_attr.append(np.concatenate([time_encoded, [interaction_type]]))
            
            edge_times.append(timestamp)
            edge_times.append(timestamp)
            
            edge_types.append(interaction_type)
            edge_types.append(interaction_type)
        
        # Create author -> post edges
        for _, row in tqdm(posts_df.iterrows(), desc="Processing authorship", 
                          total=len(posts_df)):
            if row['author_id'] in user_to_id and row['post_id'] in post_to_id:
                author_id = user_to_id[row['author_id']]
                post_id = post_to_id[row['post_id']]
                timestamp = row['timestamp'].timestamp()
                
                # Add authorship edge (type 4)
                edge_index.append((author_id, post_id))
                edge_index.append((post_id, author_id))
                
                time_encoded = self.encode_timestamps(pd.Series([row['timestamp']]))[0]
                edge_attr.append(np.concatenate([time_encoded, [4]]))
                edge_attr.append(np.concatenate([time_encoded, [4]]))
                
                edge_times.append(timestamp)
                edge_times.append(timestamp)
                
                edge_types.append(4)  # Authorship type
                edge_types.append(4)
        
        # Convert to tensor format
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
        edge_times = torch.tensor(edge_times, dtype=torch.float)
        edge_types = torch.tensor(edge_types, dtype=torch.long)
        
        # Create labels for post nodes (1 for misinformation, 0 for reliable)
        labels = torch.zeros(num_nodes, dtype=torch.long)
        for i, row in posts_df.iterrows():
            if row['post_id'] in post_to_id:
                post_idx = post_to_id[row['post_id']]
                labels[post_idx] = row['is_misinformation']
        
        # Create mapping from node IDs back to original IDs
        id_to_node = {v: k for k, v in {**user_to_id, **post_to_id}.items()}
        
        # Create node mask for posts only (for training target)
        post_mask = torch.zeros(num_nodes, dtype=torch.bool)
        for post_id, node_id in post_to_id.items():
            post_mask[node_id] = True
        
        return {
            'node_features': torch.tensor(node_features, dtype=torch.float),
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'edge_times': edge_times,
            'node_types': node_types,
            'edge_types': edge_types,
            'labels': labels,
            'id_to_node': id_to_node,
            'post_mask': post_mask,
            'num_nodes': num_nodes,
            'user_to_id': user_to_id,
            'post_to_id': post_to_id
        }
    
    def create_temporal_snapshots(self, graph_data, num_snapshots=5):
        """
        Create temporal snapshots from dynamic graph for static GCN.
        
        Args:
            graph_data (dict): Temporal graph data.
            num_snapshots (int): Number of snapshots to create.
            
        Returns:
            list: List of snapshot graphs.
        """
        edge_times = graph_data['edge_times'].numpy()
        
        # Determine time range and create snapshot intervals
        min_time = edge_times.min()
        max_time = edge_times.max()
        time_range = max_time - min_time
        snapshot_interval = time_range / num_snapshots
        
        snapshots = []
        
        for i in range(num_snapshots):
            # Define time window for this snapshot
            start_time = min_time + i * snapshot_interval
            end_time = min_time + (i + 1) * snapshot_interval if i < num_snapshots - 1 else max_time + 1
            
            # Select edges within this time window
            mask = (edge_times >= start_time) & (edge_times < end_time)
            snapshot_edges = graph_data['edge_index'][:, mask]
            snapshot_edge_attr = graph_data['edge_attr'][mask]
            snapshot_edge_types = graph_data['edge_types'][mask]
            
            snapshots.append({
                'edge_index': snapshot_edges,
                'edge_attr': snapshot_edge_attr,
                'edge_types': snapshot_edge_types,
                'time_window': (start_time, end_time)
            })
            
        return snapshots
    
    def process_dataset(self, dataset_name):
        """
        Process a single dataset and save to processed directory.
        
        Args:
            dataset_name (str): Name of the dataset to process.
            
        Returns:
            dict: Processed graph data.
        """
        posts_df, interactions_df = self.load_dataset(dataset_name)
        
        print(f"Constructing temporal graph for {dataset_name}...")
        graph_data = self.construct_temporal_graph(posts_df, interactions_df)
        
        # Create snapshots for static models
        num_snapshots = self.config['models']['gcn']['num_snapshots']
        snapshots = self.create_temporal_snapshots(graph_data, num_snapshots)
        graph_data['snapshots'] = snapshots
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        output_path = os.path.join('data/processed', f"{dataset_name}_processed.pkl")
        
        with open(output_path, 'wb') as f:
            pickle.dump(graph_data, f)
            
        print(f"Processed data saved to {output_path}")
        return graph_data
    
    def process_all_datasets(self):
        """Process all configured datasets."""
        for dataset_name in self.config['data']['datasets']:
            print(f"Processing dataset: {dataset_name}")
            self.process_dataset(dataset_name)


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.process_all_datasets()
