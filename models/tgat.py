"""
Temporal Graph Attention Network (TGAT) implementation for misinformation detection.

This model implements continuous-time dynamic graph representation learning based on
the paper "Inductive Representation Learning on Temporal Graphs" (https://arxiv.org/abs/2002.07962).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TimeEncodingLayer(nn.Module):
    """
    Time encoding layer that transforms timestamps into high-dimensional vectors.
    
    Based on the Bochner's theorem for harmonic analysis on continuous signals.
    """
    
    def __init__(self, time_dim, trainable=True, time_scale=1.0):
        """
        Initialize time encoding layer.
        
        Args:
            time_dim (int): Dimension of time encoding.
            trainable (bool): Whether the basis frequencies are trainable.
            time_scale (float): Scale factor for time values to control frequency range.
        """
        super(TimeEncodingLayer, self).__init__()
        
        self.time_dim = time_dim
        self.trainable = trainable
        self.time_scale = time_scale
        
        if trainable:
            # Create a 2D tensor for proper initialization with xavier_uniform_
            # Then reshape to 1D for the model's usage
            temp_tensor = torch.ones(time_dim, 1)
            nn.init.xavier_uniform_(temp_tensor)
            self.basis_freq = nn.Parameter(temp_tensor.view(-1))
            self.phase = nn.Parameter(torch.zeros(time_dim))
        else:
            # Fixed frequency range for Fourier basis
            freq_list = torch.logspace(start=0, end=4, steps=time_dim // 2) * time_scale
            self.register_buffer('freq_list', freq_list)
        
    def forward(self, timestamps, ref_time=None):
        """
        Forward pass: encode timestamps into trigonometric embeddings.
        
        Args:
            timestamps (torch.Tensor): Tensor of timestamps.
            ref_time (torch.Tensor, optional): Reference time for relative encoding.
                If provided, encodes t - ref_t instead of absolute time.
            
        Returns:
            torch.Tensor: Time encoding vectors.
        """
        # Apply relative time encoding if reference time is provided
        if ref_time is not None:
            if ref_time.dim() < timestamps.dim():
                ref_time = ref_time.unsqueeze(-1)
            # Compute time difference
            timestamps = timestamps - ref_time
        
        # Reshape timestamps for broadcasting
        t = timestamps.unsqueeze(-1)
        
        if self.trainable:
            # Trainable Fourier basis
            time_encoding = torch.cos(t * self.basis_freq + self.phase)
        else:
            # Fixed Fourier basis with both cos and sin components
            cos_terms = torch.cos(t * self.freq_list.unsqueeze(0))
            sin_terms = torch.sin(t * self.freq_list.unsqueeze(0))
            time_encoding = torch.cat([cos_terms, sin_terms], dim=-1)
        
        return time_encoding


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer for temporal graph attention.
    """
    
    def __init__(self, node_dim, time_dim, num_heads, attention_dim, dropout=0.1):
        """
        Initialize multi-head attention.
        
        Args:
            node_dim (int): Node feature dimension.
            time_dim (int): Time encoding dimension.
            num_heads (int): Number of attention heads.
            attention_dim (int): Dimension of each attention head.
            dropout (float): Dropout probability.
        """
        super(MultiHeadAttention, self).__init__()
        
        self.node_dim = node_dim
        self.time_dim = time_dim
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.dropout = dropout
        
        # Combined feature dimension (node features + time encoding)
        combined_dim = node_dim + time_dim
        
        # Query, key, value projections for all heads
        self.q_linear = nn.Linear(combined_dim, num_heads * attention_dim)
        self.k_linear = nn.Linear(combined_dim, num_heads * attention_dim)
        self.v_linear = nn.Linear(combined_dim, num_heads * attention_dim)
        
        # Output projection
        self.output = nn.Linear(num_heads * attention_dim, node_dim)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Scaling factor for dot-product attention
        self.scaling = attention_dim ** -0.5
        
    def forward(self, query_nodes, key_nodes, query_time, key_time, mask=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            query_nodes (torch.Tensor): Query node features [batch_size, node_dim].
            key_nodes (torch.Tensor): Key node features [batch_size, num_neighbors, node_dim].
            query_time (torch.Tensor): Query time encodings [batch_size, time_dim].
            key_time (torch.Tensor): Key time encodings [batch_size, num_neighbors, time_dim].
            mask (torch.Tensor, optional): Attention mask for padding [batch_size, num_neighbors].
            
        Returns:
            torch.Tensor: Updated node representations.
        """
        batch_size = query_nodes.size(0)
        num_neighbors = key_nodes.size(1) if len(key_nodes.size()) > 2 else 1
        
        # Combine node features with time encodings
        query_combined = torch.cat([query_nodes, query_time], dim=-1)
        
        # Reshape key nodes and time for proper concatenation
        if len(key_nodes.size()) == 3:
            # Shape: [batch, neighbors, dim]
            key_combined = torch.cat([key_nodes, key_time], dim=-1)
        else:
            # Shape: [batch, dim] - expand for single neighbor case
            key_nodes = key_nodes.unsqueeze(1)
            key_time = key_time.unsqueeze(1) if len(key_time.size()) == 2 else key_time
            key_combined = torch.cat([key_nodes, key_time], dim=-1)
        
        # Linear projections
        q = self.q_linear(query_combined)  # [batch, q_len, num_heads * attention_dim]
        k = self.k_linear(key_combined)    # [batch, k_len, num_heads * attention_dim]
        v = self.v_linear(key_combined)    # [batch, v_len, num_heads * attention_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, 1, self.num_heads, self.attention_dim)
        k = k.view(batch_size, num_neighbors, self.num_heads, self.attention_dim)
        v = v.view(batch_size, num_neighbors, self.num_heads, self.attention_dim)
        
        # Transpose to [batch, head, seq_len, attention_dim]
        q = q.transpose(1, 2)  # [batch, head, 1, attention_dim]
        k = k.transpose(1, 2)  # [batch, head, neighbors, attention_dim]
        v = v.transpose(1, 2)  # [batch, head, neighbors, attention_dim]
        
        # Compute attention scores with scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and combine heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, 1, self.num_heads * self.attention_dim)
        
        # Apply output projection
        output = self.output(attn_output.squeeze(1))
        
        return output


class TemporalGraphAttentionLayer(nn.Module):
    """
    Temporal graph attention layer that incorporates time information.
    """
    
    def __init__(self, node_dim, time_dim, num_heads, attention_dim, dropout=0.1):
        """
        Initialize temporal graph attention layer.
        
        Args:
            node_dim (int): Node feature dimension.
            time_dim (int): Time encoding dimension.
            num_heads (int): Number of attention heads.
            attention_dim (int): Dimension of each attention head.
            dropout (float): Dropout probability.
        """
        super(TemporalGraphAttentionLayer, self).__init__()
        
        self.node_dim = node_dim
        self.time_dim = time_dim
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        
        # Multi-head attention mechanism
        self.attention = MultiHeadAttention(
            node_dim=node_dim,
            time_dim=time_dim,
            num_heads=num_heads,
            attention_dim=attention_dim,
            dropout=dropout
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(node_dim, node_dim * 4),  # Wider FFN for more expressivity
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim * 4, node_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, node_features, neighbor_features, node_time, neighbor_time, mask=None):
        """
        Forward pass of temporal graph attention layer.
        
        Args:
            node_features (torch.Tensor): Features of target nodes [batch_size, node_dim].
            neighbor_features (torch.Tensor): Features of neighbor nodes [batch_size, num_neighbors, node_dim].
            node_time (torch.Tensor): Time encodings of target nodes [batch_size, time_dim].
            neighbor_time (torch.Tensor): Time encodings of interactions [batch_size, num_neighbors, time_dim].
            mask (torch.Tensor, optional): Mask for padding neighbors [batch_size, num_neighbors].
            
        Returns:
            torch.Tensor: Updated node representations.
        """
        # Multi-head attention
        attn_output = self.attention(node_features, neighbor_features, node_time, neighbor_time, mask)
        
        # Add & norm (residual connection and layer normalization)
        x = self.norm1(node_features + self.dropout(attn_output))
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        
        # Add & norm
        output = self.norm2(x + self.dropout(ff_output))
        
        return output


class TemporalGraphAttentionNetwork(nn.Module):
    """
    Temporal Graph Attention Network (TGAT) model for misinformation detection.
    
    This model uses temporal graph attention mechanisms to learn from
    dynamic temporal graphs for early misinformation detection.
    """
    
    def __init__(self, config):
        """
        Initialize the TGAT model.
        
        Args:
            config (dict): Model configuration.
        """
        super(TemporalGraphAttentionNetwork, self).__init__()
        
        self.config = config
        
        # Model dimensions
        self.node_dim = config['node_dim']
        self.time_dim = config['time_dim']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.attention_dim = config['attention_dim']
        
        # Time encoding parameters
        self.time_trainable = config.get('time_trainable', True)
        self.time_scale = config.get('time_scale', 1.0)
        
        # Time encoding layer
        self.time_encoder = TimeEncodingLayer(
            time_dim=self.time_dim,
            trainable=self.time_trainable,
            time_scale=self.time_scale
        )
        
        # Node feature projection (optional)
        if 'input_projection' in config and config['input_projection']:
            self.feature_projection = nn.Linear(self.node_dim, self.node_dim)
        else:
            self.feature_projection = nn.Identity()
        
        # Graph attention layers
        self.layers = nn.ModuleList([
            TemporalGraphAttentionLayer(
                node_dim=self.node_dim,
                time_dim=self.time_dim,
                num_heads=self.num_heads,
                attention_dim=self.attention_dim,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])
        
        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.node_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def compute_temporal_attention(self, node_features, edge_index, edge_attr, edge_times):
        """
        Compute embeddings using temporal graph attention.
        
        Args:
            node_features (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Edge connectivity.
            edge_attr (torch.Tensor): Edge features.
            edge_times (torch.Tensor): Edge timestamps.
            
        Returns:
            torch.Tensor: Node embeddings with temporal attention.
        """
        device = node_features.device
        num_nodes = node_features.size(0)
        src, dst = edge_index
        
        # Create a dictionary of node neighbors and their interaction times
        neighbors = {}
        neighbor_times = {}
        
        # Group edges by destination node
        for i in range(len(dst)):
            d = dst[i].item()
            s = src[i].item()
            t = edge_times[i].item()
            
            if d not in neighbors:
                neighbors[d] = []
                neighbor_times[d] = []
            
            neighbors[d].append(s)
            neighbor_times[d].append(t)
        
        # Initialize node embeddings with input features
        embeddings = node_features.clone()
        
        # Maximum number of neighbors to consider
        max_neighbors = 20
        
        # Process nodes in batches
        batch_size = 128
        
        # For each layer in the network
        for layer in self.layers:
            new_embeddings = embeddings.clone()
            
            # Process nodes in batches
            for start_idx in range(0, num_nodes, batch_size):
                end_idx = min(start_idx + batch_size, num_nodes)
                batch_nodes = list(range(start_idx, end_idx))
                
                # Collect neighbor information for each node in batch
                batch_neighbor_indices = []
                batch_neighbor_times = []
                batch_masks = []
                
                for node_idx in batch_nodes:
                    # Get neighbors and interaction times
                    if node_idx in neighbors:
                        node_neighbors = neighbors[node_idx]
                        node_times = neighbor_times[node_idx]
                        
                        # Sort by time (latest interactions first)
                        sorted_indices = np.argsort(node_times)[::-1]
                        node_neighbors = [node_neighbors[i] for i in sorted_indices[:max_neighbors]]
                        node_times = [node_times[i] for i in sorted_indices[:max_neighbors]]
                        
                        # Pad if fewer than max_neighbors
                        if len(node_neighbors) < max_neighbors:
                            pad_length = max_neighbors - len(node_neighbors)
                            node_neighbors.extend([0] * pad_length)
                            node_times.extend([0.0] * pad_length)
                            mask = [1] * len(sorted_indices[:max_neighbors]) + [0] * pad_length
                        else:
                            mask = [1] * max_neighbors
                    else:
                        # No neighbors, use padding
                        node_neighbors = [0] * max_neighbors
                        node_times = [0.0] * max_neighbors
                        mask = [0] * max_neighbors
                    
                    batch_neighbor_indices.append(node_neighbors)
                    batch_neighbor_times.append(node_times)
                    batch_masks.append(mask)
                
                # Convert to tensors
                batch_neighbor_indices = torch.tensor(batch_neighbor_indices, device=device)
                batch_neighbor_times = torch.tensor(batch_neighbor_times, device=device)
                batch_masks = torch.tensor(batch_masks, device=device).bool()
                
                # Get node features
                batch_node_features = embeddings[batch_nodes]
                
                # Get neighbor features
                batch_neighbor_features = embeddings[batch_neighbor_indices]
                
                # Encode times (for both nodes and neighbors)
                # Use current time as reference for query nodes (we're making predictions at the current time)
                current_time = edge_times.max().item()
                batch_node_times = torch.full((len(batch_nodes),), current_time, device=device)
                
                # Encode times using the time encoder
                batch_node_time_encodings = self.time_encoder(batch_node_times)
                batch_neighbor_time_encodings = self.time_encoder(batch_neighbor_times)
                
                # Apply temporal graph attention
                updated_features = layer(
                    batch_node_features,
                    batch_neighbor_features,
                    batch_node_time_encodings,
                    batch_neighbor_time_encodings,
                    batch_masks
                )
                
                # Update embeddings for the batch
                new_embeddings[batch_nodes] = updated_features
            
            # Update all embeddings after this layer
            embeddings = new_embeddings
        
        return embeddings
    
    def forward(self, data):
        """
        Forward pass of the TGAT model.
        
        Args:
            data (dict): Graph data containing:
                - node_features: Node feature matrix
                - edge_index: Edge connectivity
                - edge_attr: Edge features
                - edge_times: Edge timestamps
                - post_mask: Mask for post nodes
                
        Returns:
            torch.Tensor: Misinformation probability scores for post nodes.
        """
        node_features = data['node_features']
        edge_index = data['edge_index']
        edge_attr = data['edge_attr']
        edge_times = data['edge_times']
        post_mask = data['post_mask']
        
        # Compute node embeddings with temporal attention
        embeddings = self.compute_temporal_attention(
            node_features, edge_index, edge_attr, edge_times
        )
        
        # Apply classifier to post nodes only
        post_embeddings = embeddings[post_mask]
        predictions = self.classifier(post_embeddings).squeeze(-1)
        
        return predictions
