#!/usr/bin/env python
"""
Synthetic Dataset Generator for Misinformation Detection

This script generates synthetic datasets that mimic real-world misinformation
propagation patterns for testing graph-based misinformation detection models.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pickle

def generate_synthetic_dataset(
    name="synth_data",
    num_users=100,
    num_posts=50,
    num_interactions=1000,
    misinformation_ratio=0.4,
    temporal_days=7,
    user_features_dim=10,
    post_features_dim=20,
    seed=42
):
    """
    Generate a synthetic dataset with temporal interaction patterns.
    
    Args:
        name: Name of the dataset
        num_users: Number of users to generate
        num_posts: Number of posts to generate
        num_interactions: Number of interactions between users and posts
        misinformation_ratio: Ratio of misinformation posts
        temporal_days: Number of days to spread interactions over
        user_features_dim: Dimension of user features
        post_features_dim: Dimension of post features
        seed: Random seed for reproducibility
        
    Returns:
        posts_df: DataFrame of generated posts
        interactions_df: DataFrame of generated interactions
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"Generating synthetic dataset '{name}'...")
    
    # Create directory structure
    os.makedirs(f'data/raw/{name}', exist_ok=True)
    
    # Generate users
    user_ids = [f"user{i}" for i in range(num_users)]
    
    # Assign user characteristics
    user_features = np.random.normal(0, 1, (num_users, user_features_dim))
    
    # Generate posts with temporal distribution
    post_ids = [f"post{i}" for i in range(num_posts)]
    
    # Assign post characteristics
    post_features = np.random.normal(0, 1, (num_posts, post_features_dim))
    
    # Determine which posts are misinformation
    num_misinfo = int(num_posts * misinformation_ratio)
    is_misinformation = np.zeros(num_posts, dtype=int)
    is_misinformation[:num_misinfo] = 1
    np.random.shuffle(is_misinformation)
    
    # Create a base time and distribute posts over time
    base_time = datetime(2023, 1, 1, 0, 0, 0)
    post_timestamps = [base_time + timedelta(seconds=random.randint(0, 60*60*24*temporal_days)) 
                      for _ in range(num_posts)]
    
    # Create posts DataFrame
    posts_data = {
        'post_id': post_ids,
        'author_id': [random.choice(user_ids) for _ in range(num_posts)],
        'timestamp': post_timestamps,
        'content': [f"Content for post {i}" for i in range(num_posts)],
        'is_misinformation': is_misinformation
    }
    posts_df = pd.DataFrame(posts_data)
    
    # Sort posts by timestamp
    posts_df = posts_df.sort_values('timestamp')
    
    # Generate interactions with temporal patterns
    interactions = []
    
    # Model different spreading patterns for misinfo vs regular posts
    for i, post in posts_df.iterrows():
        post_id = post['post_id']
        post_time = post['timestamp']
        is_misinfo = post['is_misinformation']
        
        # Calculate how many interactions this post should have
        # Misinformation tends to spread more rapidly but among fewer credible sources
        post_interactions = int(np.random.gamma(
            shape=8 if is_misinfo else 4,  # Misinformation spreads faster initially
            scale=2 if is_misinfo else 3,  # But regular content has a longer tail
            size=1
        )[0])
        
        # Cap the interactions to ensure we don't exceed total
        post_interactions = min(post_interactions, num_interactions // num_posts * 2)
        
        # Generate the interactions for this post
        for _ in range(post_interactions):
            # For misinformation, interactions come more quickly
            if is_misinfo:
                time_delta = np.random.exponential(scale=0.5) * 86400  # in seconds, faster spread
            else:
                time_delta = np.random.gamma(shape=2, scale=0.5) * 86400  # in seconds, more gradual
                
            interaction_time = post_time + timedelta(seconds=time_delta)
            
            # Don't go beyond our temporal window
            if interaction_time > base_time + timedelta(days=temporal_days):
                continue
                
            # Get a user for this interaction (weighted sampling could be used here)
            user_id = random.choice(user_ids)
            
            # Get interaction type
            interaction_types = ['like', 'share', 'comment']
            if is_misinfo:
                # Misinformation tends to get more shares
                type_weights = [0.3, 0.5, 0.2]
            else:
                # Regular content gets more likes
                type_weights = [0.5, 0.2, 0.3]
                
            interaction_type = random.choices(interaction_types, weights=type_weights, k=1)[0]
            
            interactions.append({
                'user_id': user_id,
                'post_id': post_id,
                'timestamp': interaction_time,
                'interaction_type': interaction_type
            })
    
    # Create interactions DataFrame
    interactions_df = pd.DataFrame(interactions)
    
    # Sort interactions by timestamp
    interactions_df = interactions_df.sort_values('timestamp')
    
    # Cap to the requested number of interactions
    if len(interactions_df) > num_interactions:
        interactions_df = interactions_df.head(num_interactions)
    
    # Save dataset to disk
    posts_df.to_csv(f'data/raw/{name}/posts.csv', index=False)
    interactions_df.to_csv(f'data/raw/{name}/interactions.csv', index=False)
    
    print(f"Generated {len(posts_df)} posts and {len(interactions_df)} interactions")
    print(f"Misinformation posts: {posts_df['is_misinformation'].sum()} ({posts_df['is_misinformation'].mean()*100:.1f}%)")
    
    # Create and save a simple visualization of the dataset
    G = nx.Graph()
    
    # Add nodes
    for user_id in user_ids:
        G.add_node(user_id, type='user')
        
    for _, post in posts_df.iterrows():
        G.add_node(post['post_id'], type='post', is_misinfo=post['is_misinformation'])
    
    # Add edges
    for _, interaction in interactions_df.iterrows():
        G.add_edge(interaction['user_id'], interaction['post_id'], 
                  type=interaction['interaction_type'],
                  timestamp=interaction['timestamp'])
    
    # Draw the graph
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=seed)
    
    # Draw nodes
    user_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'user']
    post_true_nodes = [n for n, attr in G.nodes(data=True) 
                     if attr.get('type') == 'post' and attr.get('is_misinfo') == 0]
    post_misinfo_nodes = [n for n, attr in G.nodes(data=True) 
                        if attr.get('type') == 'post' and attr.get('is_misinfo') == 1]
    
    nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_color='skyblue', 
                          node_size=100, alpha=0.8, label='Users')
    nx.draw_networkx_nodes(G, pos, nodelist=post_true_nodes, node_color='green', 
                          node_size=200, alpha=0.8, label='True Posts')
    nx.draw_networkx_nodes(G, pos, nodelist=post_misinfo_nodes, node_color='red', 
                          node_size=200, alpha=0.8, label='Misinfo Posts')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    
    plt.title(f'Synthetic Dataset: {name} - {num_users} users, {num_posts} posts, {len(interactions_df)} interactions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'data/raw/{name}/graph_visualization.png', dpi=300)
    plt.close()
    
    return posts_df, interactions_df

def generate_time_series_visualization(name, posts_df, interactions_df):
    """Generate time series visualization of post creation and interactions."""
    plt.figure(figsize=(12, 6))
    
    # Convert to timestamps
    post_times = pd.to_datetime(posts_df['timestamp'])
    interaction_times = pd.to_datetime(interactions_df['timestamp'])
    
    # Group by day
    posts_per_day = post_times.groupby(post_times.dt.date).count()
    interactions_per_day = interaction_times.groupby(interaction_times.dt.date).count()
    
    # Plot
    plt.subplot(1, 2, 1)
    posts_per_day.plot(kind='bar')
    plt.title('Posts per Day')
    plt.ylabel('Count')
    plt.tight_layout()
    
    plt.subplot(1, 2, 2)
    interactions_per_day.plot(kind='bar')
    plt.title('Interactions per Day')
    plt.ylabel('Count')
    plt.tight_layout()
    
    plt.savefig(f'data/raw/{name}/time_series.png', dpi=300)
    plt.close()
    
    # Analyze misinformation spread
    misinfo_posts = posts_df[posts_df['is_misinformation'] == 1]['post_id'].tolist()
    regular_posts = posts_df[posts_df['is_misinformation'] == 0]['post_id'].tolist()
    
    misinfo_interactions = interactions_df[interactions_df['post_id'].isin(misinfo_posts)]
    regular_interactions = interactions_df[interactions_df['post_id'].isin(regular_posts)]
    
    # Group by type
    misinfo_by_type = misinfo_interactions['interaction_type'].value_counts()
    regular_by_type = regular_interactions['interaction_type'].value_counts()
    
    # Plot interaction types
    plt.figure(figsize=(10, 6))
    width = 0.35
    ind = np.arange(3)  # 3 types: like, share, comment
    
    all_types = ['like', 'share', 'comment']
    misinfo_counts = [misinfo_by_type.get(t, 0) for t in all_types]
    regular_counts = [regular_by_type.get(t, 0) for t in all_types]
    
    # Convert to percentages
    misinfo_pct = 100 * np.array(misinfo_counts) / sum(misinfo_counts)
    regular_pct = 100 * np.array(regular_counts) / sum(regular_counts)
    
    plt.bar(ind - width/2, misinfo_pct, width, label='Misinformation Posts')
    plt.bar(ind + width/2, regular_pct, width, label='Regular Posts')
    
    plt.ylabel('Percentage')
    plt.title('Interaction Types by Post Category')
    plt.xticks(ind, all_types)
    plt.legend()
    
    plt.savefig(f'data/raw/{name}/interaction_types.png', dpi=300)
    plt.close()

def main():
    """Generate multiple synthetic datasets of varying sizes."""
    # Small dataset similar to CoAID
    posts_df, interactions_df = generate_synthetic_dataset(
        name="synth_small",
        num_users=15,
        num_posts=5,
        num_interactions=30,
        misinformation_ratio=0.6,
        temporal_days=1,
        seed=42
    )
    generate_time_series_visualization("synth_small", posts_df, interactions_df)
    
    # Medium dataset (more realistic size)
    posts_df, interactions_df = generate_synthetic_dataset(
        name="synth_medium",
        num_users=500,
        num_posts=200,
        num_interactions=5000,
        misinformation_ratio=0.4,
        temporal_days=14,
        seed=43
    )
    generate_time_series_visualization("synth_medium", posts_df, interactions_df)
    
    # Large dataset for scaling tests
    posts_df, interactions_df = generate_synthetic_dataset(
        name="synth_large",
        num_users=2000,
        num_posts=1000,
        num_interactions=50000,
        misinformation_ratio=0.3,
        temporal_days=30,
        seed=44
    )
    generate_time_series_visualization("synth_large", posts_df, interactions_df)
    
    print("All synthetic datasets generated successfully!")

if __name__ == "__main__":
    main() 