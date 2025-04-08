#!/usr/bin/env python
"""
Full Dataset Preprocessing Script 

This script processes the complete CoAID and FakeNewsNet datasets to create 
temporal interaction graphs for misinformation detection.
"""

import os
import pandas as pd
import numpy as np
import glob
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import networkx as nx
import torch
from tqdm import tqdm
import yaml

# Ensure NLTK resources are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

class FullDatasetPreprocessor:
    """
    Preprocessor for the full CoAID and FakeNewsNet datasets.
    """
    
    def __init__(self, config_path='configs/config.yaml'):
        """Initialize the preprocessor with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize text vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config['data']['feature_engineering']['text']['max_features'],
            stop_words='english'
        )
    
    def clean_text(self, text):
        """Clean text by removing special characters and lowercasing."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_text(self, text_series):
        """Preprocess text data for feature extraction."""
        processed_texts = []
        
        for text in text_series:
            if not isinstance(text, str):
                processed_texts.append('')
                continue
                
            # Clean text
            text = self.clean_text(text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords
            tokens = [word for word in tokens if word not in self.stop_words]
            
            # Rejoin tokens
            processed_text = ' '.join(tokens)
            processed_texts.append(processed_text)
            
        return processed_texts
    
    def process_coaid(self):
        """Process the CoAID dataset."""
        print("Processing CoAID dataset...")
        
        # Paths to dataset directories
        coaid_dir = self.config['paths']['coaid']['raw']
        
        # Initialize dataframes to store posts and interactions
        all_posts = []
        all_interactions = []
        
        # Process each version directory
        version_dirs = [d for d in os.listdir(coaid_dir) 
                       if os.path.isdir(os.path.join(coaid_dir, d)) and d.endswith('-2020')]
        
        print(f"Found {len(version_dirs)} version directories: {version_dirs}")
        
        for version_dir in version_dirs:
            print(f"\nProcessing CoAID directory: {version_dir}")
            dir_path = os.path.join(coaid_dir, version_dir)
            
            # Process news files
            news_files = glob.glob(os.path.join(dir_path, 'News*.csv'))
            news_files = [f for f in news_files if not ('tweets' in f.lower() or 'replies' in f.lower())]
            print(f"Found {len(news_files)} news files")
            
            for file_path in news_files:
                try:
                    print(f"Processing news file: {os.path.basename(file_path)}")
                    df = pd.read_csv(file_path)
                    
                    # Determine if it's fake or real news
                    is_fake = 'Fake' in file_path
                    
                    # Create posts dataframe
                    posts_df = pd.DataFrame()
                    posts_df['post_id'] = df.index.astype(str)  # Using index as ID
                    posts_df['author_id'] = df['type'].fillna('unknown')  # Using type as author
                    posts_df['timestamp'] = pd.to_datetime(df['publish_date']).fillna(pd.Timestamp.now()).dt.strftime('%Y-%m-%d %H:%M:%S')
                    posts_df['content'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
                    posts_df['is_misinformation'] = 1 if is_fake else 0
                    
                    all_posts.append(posts_df)
                    print(f"Added {len(posts_df)} news posts")
                    
                except Exception as e:
                    print(f"Error processing news file {file_path}: {str(e)}")
            
            # Process tweets and replies
            tweet_files = glob.glob(os.path.join(dir_path, '*tweets.csv'))
            tweet_files = [f for f in tweet_files if not 'replies' in f.lower()]
            reply_files = glob.glob(os.path.join(dir_path, '*replies.csv'))
            
            print(f"Found {len(tweet_files)} tweet files and {len(reply_files)} reply files")
            
            # Process tweets (posts)
            for file_path in tweet_files:
                try:
                    print(f"Processing tweet file: {os.path.basename(file_path)}")
                    df = pd.read_csv(file_path)
                    
                    # Determine if it's fake or real
                    is_fake = 'Fake' in file_path
                    
                    posts_df = pd.DataFrame()
                    posts_df['post_id'] = df['tweet_id'].astype(str)
                    posts_df['author_id'] = 'twitter_user_' + df['index'].astype(str)  # Creating synthetic user IDs
                    posts_df['timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')  # Using current time as tweets don't have timestamps
                    posts_df['content'] = 'Tweet ' + df['tweet_id'].astype(str)  # Using tweet ID as content since we don't have the actual text
                    posts_df['is_misinformation'] = 1 if is_fake else 0
                    
                    all_posts.append(posts_df)
                    print(f"Added {len(posts_df)} tweet posts")
                    
                except Exception as e:
                    print(f"Error processing tweet file {file_path}: {str(e)}")
            
            # Process replies (interactions)
            for file_path in reply_files:
                try:
                    print(f"Processing reply file: {os.path.basename(file_path)}")
                    df = pd.read_csv(file_path)
                    
                    interactions_df = pd.DataFrame()
                    interactions_df['post_id'] = df['tweet_id'].astype(str)
                    interactions_df['user_id'] = 'replier_' + df['reply_id'].astype(str)  # Creating synthetic user IDs
                    interactions_df['timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')  # Using current time as replies don't have timestamps
                    interactions_df['content'] = 'Reply ' + df['reply_id'].astype(str)  # Using reply ID as content
                    interactions_df['interaction_type'] = 'reply'
                    
                    all_interactions.append(interactions_df)
                    print(f"Added {len(interactions_df)} interactions")
                    
                except Exception as e:
                    print(f"Error processing reply file {file_path}: {str(e)}")
        
        # Combine all posts and interactions
        if all_posts:
            final_posts = pd.concat(all_posts, ignore_index=True)
            final_posts = final_posts.drop_duplicates(subset=['post_id']).reset_index(drop=True)
            print(f"\nTotal posts collected: {len(final_posts)}")
            print(f"Misinformation posts: {final_posts['is_misinformation'].sum()} ({final_posts['is_misinformation'].mean()*100:.1f}%)")
        else:
            final_posts = pd.DataFrame(columns=['post_id', 'author_id', 'timestamp', 'content', 'is_misinformation'])
            print("No posts were collected!")
        
        if all_interactions:
            final_interactions = pd.concat(all_interactions, ignore_index=True)
            final_interactions = final_interactions.drop_duplicates(subset=['post_id', 'user_id', 'timestamp']).reset_index(drop=True)
            print(f"Total interactions collected: {len(final_interactions)}")
        else:
            final_interactions = pd.DataFrame(columns=['post_id', 'user_id', 'timestamp', 'content', 'interaction_type'])
            print("No interactions were collected!")
        
        return final_posts, final_interactions
    
    def process_fakenewsnet(self):
        """Process the FakeNewsNet dataset."""
        print("Processing FakeNewsNet dataset...")
        
        # Paths to dataset files
        fakenewsnet_dir = 'data/raw/fakenewsnet'
        
        # Initialize dataframes to store posts
        all_posts = []
        
        # Process each file
        for file_name in os.listdir(fakenewsnet_dir):
            if file_name.endswith('.csv'):
                file_path = os.path.join(fakenewsnet_dir, file_name)
                print(f"Processing FakeNewsNet file: {file_name}")
                
                try:
                    # Determine if it's real or fake
                    is_fake = 1 if 'fake' in file_name.lower() else 0
                    
                    # Read data
                    df = pd.read_csv(file_path, encoding='utf-8')
                    
                    # Create posts dataframe
                    posts_df = pd.DataFrame()
                    posts_df['post_id'] = df['id'].astype(str)
                    posts_df['author_id'] = 'source_' + file_name.split('.')[0]  # Use filename as author
                    
                    # Extract timestamp if available, otherwise use current date
                    posts_df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Use title as content
                    posts_df['content'] = df['title']
                    
                    # Set misinformation flag
                    posts_df['is_misinformation'] = is_fake
                    
                    all_posts.append(posts_df)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
        
        # Combine all posts
        if all_posts:
            combined_posts = pd.concat(all_posts, ignore_index=True)
            # Remove duplicates
            combined_posts = combined_posts.drop_duplicates(subset=['post_id']).reset_index(drop=True)
            print(f"Total FakeNewsNet posts: {len(combined_posts)}")
            print(f"Misinformation posts: {combined_posts['is_misinformation'].sum()} ({combined_posts['is_misinformation'].mean()*100:.1f}%)")
        else:
            combined_posts = pd.DataFrame(columns=['post_id', 'author_id', 'timestamp', 'content', 'is_misinformation'])
            print("No posts found in FakeNewsNet dataset")
        
        # For FakeNewsNet, we don't have direct interactions, so we'll leave it empty
        combined_interactions = pd.DataFrame(columns=['timestamp', 'user_id', 'post_id', 'interaction_type', 'content'])
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        
        # Save to CSV files for later use
        combined_posts.to_csv('data/processed/fakenewsnet_posts.csv', index=False)
        combined_interactions.to_csv('data/processed/fakenewsnet_interactions.csv', index=False)
        
        return combined_posts, combined_interactions
    
    def construct_temporal_graph(self, posts_df, interactions_df, dataset_name):
        """
        Construct a temporal graph from processed posts and interactions.
        
        Args:
            posts_df: DataFrame containing post information
            interactions_df: DataFrame containing interaction information
            dataset_name: Name of the dataset
            
        Returns:
            dict: Processed graph data
        """
        print(f"Constructing temporal graph for {dataset_name}...")
        
        # Convert timestamps to datetime
        try:
            posts_df['timestamp'] = pd.to_datetime(posts_df['timestamp'])
        except:
            print("Warning: Could not parse post timestamps. Using current time.")
            posts_df['timestamp'] = datetime.now()
            
        try:
            interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
        except:
            print("Warning: Could not parse interaction timestamps. Using current time.")
            interactions_df['timestamp'] = datetime.now()
        
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
        node_dim = self.config['models']['tgn']['node_dim']
        node_features = torch.zeros((num_nodes, node_dim))
        
        # Extract text features for post nodes
        if len(posts_df) > 0:
            processed_texts = self.preprocess_text(posts_df['content'])
            
            # Create TF-IDF features
            if len(processed_texts) > 0:
                try:
                    text_features = self.tfidf_vectorizer.fit_transform(processed_texts).toarray()
                    
                    # Assign text features to post nodes
                    for i, post_id in enumerate(posts_df['post_id']):
                        if post_id in post_to_id:  # Check if post exists in mapping
                            node_idx = post_to_id[post_id]
                            feature_dim = min(text_features.shape[1], node_features.shape[1])
                            node_features[node_idx, :feature_dim] = torch.tensor(text_features[i, :feature_dim], dtype=torch.float)
                except Exception as e:
                    print(f"Error creating text features: {str(e)}")
        
        # Create edges from interactions
        edge_index = []
        edge_attr = []
        edge_times = []
        edge_types = []
        
        # Add edges for each interaction
        for _, row in tqdm(interactions_df.iterrows(), desc="Processing interactions", 
                          total=len(interactions_df)):
            if row['user_id'] in user_to_id and row['post_id'] in post_to_id:
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
                
                # Add time encoding
                time_encoded = torch.zeros(self.config['data']['feature_engineering']['time']['encoding_dim'])
                for i in range(self.config['data']['feature_engineering']['time']['encoding_dim'] // 2):
                    omega = 1.0 / (10000 ** (2 * i / self.config['data']['feature_engineering']['time']['encoding_dim']))
                    time_encoded[2*i] = torch.sin(torch.tensor(omega * timestamp))
                    time_encoded[2*i+1] = torch.cos(torch.tensor(omega * timestamp))
                
                # Add edge features and metadata for both directions
                edge_attr.append(torch.cat([time_encoded, torch.tensor([interaction_type])]))
                edge_attr.append(torch.cat([time_encoded, torch.tensor([interaction_type])]))
                
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
                
                try:
                    timestamp = row['timestamp'].timestamp()
                except:
                    timestamp = datetime.now().timestamp()
                
                # Add authorship edge (type 4)
                edge_index.append((author_id, post_id))
                edge_index.append((post_id, author_id))
                
                # Add time encoding
                time_encoded = torch.zeros(self.config['data']['feature_engineering']['time']['encoding_dim'])
                for i in range(self.config['data']['feature_engineering']['time']['encoding_dim'] // 2):
                    omega = 1.0 / (10000 ** (2 * i / self.config['data']['feature_engineering']['time']['encoding_dim']))
                    time_encoded[2*i] = torch.sin(torch.tensor(omega * timestamp))
                    time_encoded[2*i+1] = torch.cos(torch.tensor(omega * timestamp))
                
                edge_attr.append(torch.cat([time_encoded, torch.tensor([4])]))
                edge_attr.append(torch.cat([time_encoded, torch.tensor([4])]))
                
                edge_times.append(timestamp)
                edge_times.append(timestamp)
                
                edge_types.append(4)  # Authorship type
                edge_types.append(4)
        
        # Handle empty graphs
        if not edge_index:
            print(f"Warning: No edges in the {dataset_name} graph. Creating dummy edge.")
            # Create a dummy edge to prevent errors
            edge_index = [(0, 1), (1, 0)]
            time_encoded = torch.zeros(self.config['data']['feature_engineering']['time']['encoding_dim'])
            edge_attr = [
                torch.cat([time_encoded, torch.tensor([0])]),
                torch.cat([time_encoded, torch.tensor([0])])
            ]
            edge_times = [0, 0]
            edge_types = [0, 0]
        
        # Convert to tensor format
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr)
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
        
        print(f"Graph construction complete: {num_nodes} nodes, {edge_index.shape[1]} edges")
        
        # Create temporal snapshots
        num_snapshots = self.config['models']['gcn']['num_snapshots']
        min_time = edge_times.min().item()
        max_time = edge_times.max().item()
        time_range = max_time - min_time
        snapshot_interval = time_range / num_snapshots
        
        snapshots = []
        for i in range(num_snapshots):
            # Define time window for this snapshot
            start_time = min_time + i * snapshot_interval
            end_time = min_time + (i + 1) * snapshot_interval if i < num_snapshots - 1 else max_time + 1
            
            # Select edges within this time window
            mask = (edge_times >= start_time) & (edge_times < end_time)
            snapshot_edges = edge_index[:, mask]
            snapshot_edge_attr = edge_attr[mask]
            snapshot_edge_types = edge_types[mask]
            
            snapshots.append({
                'edge_index': snapshot_edges,
                'edge_attr': snapshot_edge_attr,
                'edge_types': snapshot_edge_types,
                'time_window': (start_time, end_time)
            })
        
        # Return graph data
        graph_data = {
            'node_features': node_features,
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
            'post_to_id': post_to_id,
            'snapshots': snapshots
        }
        
        # Save processed data
        output_path = os.path.join('data/processed', f"{dataset_name}_processed.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(graph_data, f)
            
        print(f"Processed data saved to {output_path}")
        
        return graph_data
    
    def process_all_datasets(self):
        """Process all datasets."""
        # Process CoAID dataset
        print("\nProcessing CoAID dataset...")
        coaid_posts, coaid_interactions = self.process_coaid()
        
        if len(coaid_posts) > 0:
            print("\nConstructing temporal graph for CoAID...")
            coaid_graph = self.construct_temporal_graph(coaid_posts, coaid_interactions, 'coaid')
            
            # Save processed data
            print("\nSaving processed data...")
            processed_dir = os.path.dirname(self.config['paths']['coaid']['processed'])
            os.makedirs(processed_dir, exist_ok=True)
            
            with open(self.config['paths']['coaid']['processed'], 'wb') as f:
                pickle.dump(coaid_graph, f)
            
            print(f"CoAID data saved to {self.config['paths']['coaid']['processed']}")
        else:
            print("No CoAID data to process!")
            coaid_graph = None
        
        return {'coaid': coaid_graph}

def main():
    """Main function to process the CoAID dataset."""
    print("Starting data preprocessing...")
    preprocessor = FullDatasetPreprocessor()
    graphs = preprocessor.process_all_datasets()
    print("\nPreprocessing completed!")

if __name__ == "__main__":
    main() 