#!/usr/bin/env python
"""
Utility script to create a synthetic dataset for testing the misinformation detection pipeline.

This script generates:
1. A posts.csv file with synthetic post content and binary labels
2. An interactions.csv file with user-post interactions
"""

import os
import pandas as pd
import numpy as np
import random
import argparse
from datetime import datetime, timedelta
import string

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Create synthetic dataset for testing')
    parser.add_argument('--dataset', type=str, default='test_dataset',
                        help='Name of the dataset to create')
    parser.add_argument('--num_posts', type=int, default=100,
                        help='Number of posts to generate')
    parser.add_argument('--num_users', type=int, default=50,
                        help='Number of users to generate')
    parser.add_argument('--num_interactions', type=int, default=500,
                        help='Number of interactions to generate')
    parser.add_argument('--misinformation_ratio', type=float, default=0.3,
                        help='Ratio of misinformation posts (0-1)')
    parser.add_argument('--time_span_days', type=int, default=30,
                        help='Time span for data generation in days')
    return parser.parse_args()

def generate_random_text(min_words=5, max_words=30):
    """Generate random text for post content."""
    num_words = random.randint(min_words, max_words)
    words = []
    for _ in range(num_words):
        word_length = random.randint(3, 10)
        word = ''.join(random.choice(string.ascii_lowercase) for _ in range(word_length))
        words.append(word)
    return ' '.join(words)

def create_posts_df(num_posts, misinformation_ratio, time_span_days):
    """Create a DataFrame of posts with random content and timestamps."""
    # Generate post IDs
    post_ids = list(range(1, num_posts + 1))
    
    # Generate random content
    contents = [generate_random_text() for _ in range(num_posts)]
    
    # Generate labels (0 for genuine, 1 for misinformation)
    labels = np.random.choice([0, 1], size=num_posts, 
                             p=[1-misinformation_ratio, misinformation_ratio])
    
    # Generate timestamps over the specified time span
    end_date = datetime.now()
    start_date = end_date - timedelta(days=time_span_days)
    timestamps = [start_date + (end_date - start_date) * random.random()
                 for _ in range(num_posts)]
    timestamps.sort()  # Sort in chronological order
    
    # Convert timestamps to string format
    timestamp_strs = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps]
    
    # Create topics
    topics = ['politics', 'health', 'technology', 'entertainment', 'sports']
    post_topics = [random.choice(topics) for _ in range(num_posts)]
    
    # Create authors (users who created the posts)
    authors = [f'user{random.randint(1, 100)}' for _ in range(num_posts)]
    
    # Create DataFrame
    posts_df = pd.DataFrame({
        'post_id': post_ids,
        'content': contents,
        'timestamp': timestamp_strs,
        'label': labels,
        'topic': post_topics,
        'author': authors
    })
    
    return posts_df

def create_interactions_df(posts_df, num_users, num_interactions):
    """Create a DataFrame of user-post interactions."""
    # Generate user IDs
    user_ids = [f'user{i}' for i in range(1, num_users + 1)]
    
    # List of possible interaction types
    interaction_types = ['like', 'share', 'comment']
    
    # Create empty lists for interaction data
    interaction_user_ids = []
    interaction_post_ids = []
    interaction_timestamps = []
    interaction_types_list = []
    interaction_contents = []
    
    # Get post IDs and timestamps from posts_df
    post_ids = posts_df['post_id'].tolist()
    post_timestamps = pd.to_datetime(posts_df['timestamp'])
    
    # Generate random interactions
    for _ in range(num_interactions):
        # Select a random post
        post_idx = random.randint(0, len(post_ids) - 1)
        post_id = post_ids[post_idx]
        post_time = post_timestamps[post_idx]
        
        # Select a random user
        user_id = random.choice(user_ids)
        
        # Generate a timestamp after the post creation
        interaction_time = post_time + timedelta(
            minutes=random.randint(1, 60*24*7))  # Within a week of the post
        
        # Select a random interaction type
        interaction_type = random.choice(interaction_types)
        
        # Generate content for comments
        content = generate_random_text(2, 10) if interaction_type == 'comment' else ''
        
        # Add to lists
        interaction_user_ids.append(user_id)
        interaction_post_ids.append(post_id)
        interaction_timestamps.append(interaction_time.strftime('%Y-%m-%d %H:%M:%S'))
        interaction_types_list.append(interaction_type)
        interaction_contents.append(content)
    
    # Create DataFrame
    interactions_df = pd.DataFrame({
        'user_id': interaction_user_ids,
        'post_id': interaction_post_ids,
        'timestamp': interaction_timestamps,
        'interaction_type': interaction_types_list,
        'content': interaction_contents
    })
    
    # Sort by timestamp
    interactions_df = interactions_df.sort_values('timestamp')
    
    return interactions_df

def main():
    """Main function to create the synthetic dataset."""
    args = parse_args()
    
    print(f"Generating synthetic dataset: {args.dataset}")
    print(f"Number of posts: {args.num_posts}")
    print(f"Number of users: {args.num_users}")
    print(f"Number of interactions: {args.num_interactions}")
    print(f"Misinformation ratio: {args.misinformation_ratio}")
    print(f"Time span: {args.time_span_days} days")
    
    # Create posts DataFrame
    posts_df = create_posts_df(
        args.num_posts,
        args.misinformation_ratio,
        args.time_span_days
    )
    
    # Create interactions DataFrame
    interactions_df = create_interactions_df(
        posts_df,
        args.num_users,
        args.num_interactions
    )
    
    # Create output directory
    output_dir = os.path.join('data', 'raw', args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV files
    posts_file = os.path.join(output_dir, 'posts.csv')
    interactions_file = os.path.join(output_dir, 'interactions.csv')
    
    posts_df.to_csv(posts_file, index=False)
    interactions_df.to_csv(interactions_file, index=False)
    
    print(f"Dataset created successfully:")
    print(f"  Posts file: {posts_file} ({len(posts_df)} rows)")
    print(f"  Interactions file: {interactions_file} ({len(interactions_df)} rows)")
    
    # Print statistics
    num_misinfo = (posts_df['label'] == 1).sum()
    num_genuine = (posts_df['label'] == 0).sum()
    
    print("\nDataset Statistics:")
    print(f"  Misinformation posts: {num_misinfo} ({num_misinfo/len(posts_df):.1%})")
    print(f"  Genuine posts: {num_genuine} ({num_genuine/len(posts_df):.1%})")
    
    like_count = (interactions_df['interaction_type'] == 'like').sum()
    share_count = (interactions_df['interaction_type'] == 'share').sum()
    comment_count = (interactions_df['interaction_type'] == 'comment').sum()
    
    print(f"  Likes: {like_count} ({like_count/len(interactions_df):.1%})")
    print(f"  Shares: {share_count} ({share_count/len(interactions_df):.1%})")
    print(f"  Comments: {comment_count} ({comment_count/len(interactions_df):.1%})")

if __name__ == "__main__":
    main() 