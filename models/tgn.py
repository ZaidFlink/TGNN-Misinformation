"""
Temporal Graph Network (TGN) implementation for misinformation detection.

This model incorporates temporal dynamics and node memory for learning on
dynamic graphs. Based on the paper "Temporal Graph Networks for Deep Learning
on Dynamic Graphs" (https://arxiv.org/abs/2006.10637).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class MemoryModule(nn.Module):
    """Memory module for TGN that maintains and updates node states over time."""
    
    def __init__(self, memory_dim, node_features_dim, message_dim, device='cpu', memory_updater='gru'):
        """
        Initialize the memory module.
        
        Args:
            memory_dim (int): Dimension of memory vectors.
            node_features_dim (int): Dimension of node features.
            message_dim (int): Dimension of message vectors.
            device (str): Device to use ('cpu' or 'cuda').
            memory_updater (str): Memory update mechanism ('gru', 'rnn', or 'mlp').
        """
        super(MemoryModule, self).__init__()
        
        self.memory_dim = memory_dim
        self.node_features_dim = node_features_dim
        self.message_dim = message_dim
        self.device = device
        self.memory_updater = memory_updater
        
        # Memory updating mechanism
        if memory_updater == 'gru':
            self.memory_updater_func = nn.GRUCell(
                input_size=message_dim,
                hidden_size=memory_dim
            )
        elif memory_updater == 'rnn':
            self.memory_updater_func = nn.RNNCell(
                input_size=message_dim,
                hidden_size=memory_dim
            )
        elif memory_updater == 'mlp':
            self.memory_updater_func = nn.Sequential(
                nn.Linear(message_dim + memory_dim, memory_dim),
                nn.ReLU(),
                nn.Linear(memory_dim, memory_dim)
            )
        else:
            raise ValueError(f"Unknown memory updater: {memory_updater}")
        
        # Message function: combines node features and message content
        self.message_fn = nn.Sequential(
            nn.Linear(node_features_dim + memory_dim + message_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim)
        )
        
        # Time projection layer for temporal effects
        self.time_proj = nn.Linear(1, message_dim)
        
        # Last update time for each node
        self.last_update = None
        # Memory state for each node
        self.memory = None
        # Messages for each node
        self.messages = None
        
    def reset_state(self, num_nodes):
        """
        Reset memory state for a new batch or epoch.
        
        Args:
            num_nodes (int): Number of nodes in the graph.
        """
        self.last_update = torch.zeros(num_nodes).to(self.device)
        self.memory = torch.zeros(num_nodes, self.memory_dim).to(self.device)
        self.messages = torch.zeros(num_nodes, self.message_dim).to(self.device)
        
    def update_memory(self, node_idxs, node_features, edge_features, timestamps):
        """
        Update memory states for the given nodes.
        
        Args:
            node_idxs (torch.Tensor): Indices of nodes to update.
            node_features (torch.Tensor): Features of the nodes.
            edge_features (torch.Tensor): Features of the edges (including time encoding).
            timestamps (torch.Tensor): Timestamps of the updates.
        """
        if self.memory is None:
            max_idx = max(node_idxs.max().item() + 1, 10000)  # Ensure minimum size
            self.reset_state(max_idx)
        
        # Compute time delta for temporal effects
        prev_update = self.last_update[node_idxs]
        curr_update = timestamps
        delta_t = curr_update - prev_update
        
        # Prepare memory for message computation
        node_memory = self.memory[node_idxs]
        
        # Compute messages using node features, memory, and edge features
        # Make sure edge_features is already combined with time encoding
        messages = self.message_fn(
            torch.cat([node_features, node_memory, edge_features], dim=1)
        )
        
        # Aggregate messages for each node (using most recent message)
        self.messages[node_idxs] = messages
        
        # Update memory based on the chosen mechanism
        if self.memory_updater in ['gru', 'rnn']:
            updated_memory = self.memory_updater_func(
                messages,
                node_memory
            )
        else:  # MLP
            updated_memory = self.memory_updater_func(
                torch.cat([messages, node_memory], dim=1)
            )
        
        # Update memory and last update time
        self.memory[node_idxs] = updated_memory
        self.last_update[node_idxs] = curr_update
        
    def get_memory(self, node_idxs=None):
        """
        Retrieve memory states for specified nodes.
        
        Args:
            node_idxs (torch.Tensor, optional): Indices of nodes to retrieve.
                If None, returns memory for all nodes.
                
        Returns:
            torch.Tensor: Memory states for the requested nodes.
        """
        if node_idxs is None:
            return self.memory
        else:
            return self.memory[node_idxs]


class TimeEncoder(nn.Module):
    """Encoder for temporal information."""
    
    def __init__(self, time_dim, method='sin'):
        """
        Initialize time encoder.
        
        Args:
            time_dim (int): Dimension of time encoding.
            method (str): Encoding method ('sin', 'fourier', or 'learnable').
        """
        super(TimeEncoder, self).__init__()
        
        self.time_dim = time_dim
        self.method = method
        
        if method == 'sin':
            # Learnable parameters for time encoding
            self.w = nn.Parameter(torch.ones(time_dim))
            self.b = nn.Parameter(torch.zeros(time_dim))
        elif method == 'fourier':
            # Fourier basis frequencies
            time_scale = 1.0  # Adjustable scale for time differences
            freq_list = torch.logspace(start=0, end=9, steps=time_dim // 2) * time_scale
            self.register_buffer('freq_list', freq_list)
        elif method == 'learnable':
            # Fully learnable time encoding
            self.time_encoder = nn.Sequential(
                nn.Linear(1, time_dim),
                nn.ReLU(),
                nn.Linear(time_dim, time_dim)
            )
        else:
            raise ValueError(f"Unknown time encoding method: {method}")
        
    def forward(self, timestamps):
        """
        Encode timestamps into high-dimensional vectors.
        
        Args:
            timestamps (torch.Tensor): Tensor of timestamps.
            
        Returns:
            torch.Tensor: Encoded timestamps.
        """
        if self.method == 'sin':
            # Simple sine function for encoding
            time_encoding = torch.sin(self.w * timestamps.unsqueeze(1) + self.b)
        elif self.method == 'fourier':
            # Fourier basis encoding
            t = timestamps.unsqueeze(1)
            cos_terms = torch.cos(t * self.freq_list.unsqueeze(0))
            sin_terms = torch.sin(t * self.freq_list.unsqueeze(0))
            time_encoding = torch.cat([cos_terms, sin_terms], dim=1)
        elif self.method == 'learnable':
            # Fully learnable encoding
            time_encoding = self.time_encoder(timestamps.unsqueeze(1))
        
        return time_encoding


class GraphAttentionLayer(nn.Module):
    """Graph attention layer for node embedding."""
    
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, concat=True):
        """
        Initialize graph attention layer.
        
        Args:
            in_dim (int): Input feature dimension.
            out_dim (int): Output feature dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            concat (bool): Whether to concatenate attention heads (True) or average them (False).
        """
        super(GraphAttentionLayer, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat = concat
        
        if concat:
            self.out_dim_per_head = out_dim // num_heads
            assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        else:
            self.out_dim_per_head = out_dim
        
        # Using PyTorch Geometric's GATConv
        self.attention_layers = nn.ModuleList()
        for _ in range(num_heads):
            self.attention_layers.append(
                GATConv(in_dim, self.out_dim_per_head, dropout=dropout)
            )
        
    def forward(self, x, edge_index, edge_time=None):
        """
        Forward pass of graph attention layer.
        
        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
            edge_time (torch.Tensor, optional): Edge timestamps (not used in this layer).
            
        Returns:
            torch.Tensor: Updated node embeddings.
        """
        outputs = []
        
        for attn in self.attention_layers:
            outputs.append(attn(x, edge_index))
            
        if self.concat:
            return torch.cat(outputs, dim=1)
        else:
            outputs_tensor = torch.stack(outputs, dim=0)
            return torch.mean(outputs_tensor, dim=0)


class TemporalGraphNetwork(nn.Module):
    """
    Temporal Graph Network (TGN) for dynamic graph learning.
    
    Adapts the architecture from the paper "Temporal Graph Networks for Deep Learning on Dynamic Graphs"
    (https://arxiv.org/abs/2006.10637)
    """
    
    def __init__(self, 
                 node_dim, 
                 edge_dim,
                 time_dim,
                 memory_dim,
                 embedding_dim=32,  # Reduced from default
                 num_neighbors=5,   # Reduced for small graph
                 num_layers=1,      # Reduced number of layers
                 dropout=0.1,
                 use_memory=True,
                 memory_updater='rnn',
                 memory_update_at_end=True,
                 embedding_module='graph_attention',
                 num_classes=2):
        """
        Initialize TGN model.
        
        Args:
            node_dim (int): Dimension of input node features.
            edge_dim (int): Dimension of input edge features.
            time_dim (int): Dimension of time encoding.
            memory_dim (int): Dimension of memory.
            embedding_dim (int): Dimension of output embeddings.
            num_neighbors (int): Number of neighbors to sample.
            num_layers (int): Number of layers.
            dropout (float): Dropout rate.
            use_memory (bool): Whether to use memory.
            memory_updater (str): Type of memory updater ('rnn', 'gru').
            memory_update_at_end (bool): Whether to update memory at the end of batch.
            embedding_module (str): Type of embedding module ('graph_attention', 'graph_sum', 'identity').
            num_classes (int): Number of output classes.
        """
        super(TemporalGraphNetwork, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim
        self.num_neighbors = num_neighbors
        self.num_layers = num_layers
        self.use_memory = use_memory
        self.memory_update_at_end = memory_update_at_end
        
        # Initialize memory for nodes
        if self.use_memory:
            self.memory_dim = memory_dim
            
            # Memory updater
            memory_input_dim = node_dim + memory_dim + edge_dim
            
            # Message function to generate messages for memory update
            self.message_fn = torch.nn.Sequential(
                torch.nn.Linear(memory_input_dim, embedding_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(embedding_dim, memory_dim),
                torch.nn.ReLU()
            )
            
            # Memory updater module
            if memory_updater == 'rnn':
                self.memory_updater = MemoryUpdater(memory_dim, memory_dim)
            elif memory_updater == 'gru':
                self.memory_updater = torch.nn.GRUCell(memory_dim, memory_dim)
            else:
                raise ValueError(f"Unknown memory updater: {memory_updater}")
            
            # Time encoder for memory update
            self.time_encoder = TimeEncoder(time_dim)
            
            # Initialize memory
            self.memory = Memory(memory_dim)
        
        # Embedding module
        if embedding_module == 'graph_attention':
            self.embedding_module = GraphAttentionEmbedding(
                node_dim + memory_dim if use_memory else node_dim,
                embedding_dim,
                edge_dim + time_dim,
                num_neighbors,
                dropout
            )
        elif embedding_module == 'graph_sum':
            self.embedding_module = GraphSumEmbedding(
                node_dim + memory_dim if use_memory else node_dim,
                embedding_dim,
                edge_dim + time_dim
            )
        elif embedding_module == 'identity':
            self.embedding_module = IdentityEmbedding(
                node_dim + memory_dim if use_memory else node_dim,
                embedding_dim
            )
        else:
            raise ValueError(f"Unknown embedding module: {embedding_module}")
        
        # MLP for final prediction
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(embedding_dim // 2, num_classes)
        )

    def compute_temporal_embeddings(self, node_features, edge_index, edge_attr, edge_times):
        """
        Compute embeddings for all nodes considering temporal information.
        
        Args:
            node_features (torch.Tensor): Features of all nodes.
            edge_index (torch.Tensor): Edge indices.
            edge_attr (torch.Tensor): Edge features.
            edge_times (torch.Tensor): Edge timestamps.
            
        Returns:
            torch.Tensor: Temporal embeddings for all nodes.
        """
        # Process edges in temporal order
        _, time_order = torch.sort(edge_times)
        edge_index_t = edge_index[:, time_order]
        edge_attr_t = edge_attr[time_order]
        edge_times_t = edge_times[time_order]
        
        # Reset memory at the beginning of forward pass
        self.memory.reset_state(self.num_nodes)
        
        # Initialize embeddings
        embeddings = node_features.clone()
        
        # Process edges in batches for memory updates
        batch_size = 200  # Adjust based on memory constraints
        num_edges = edge_index_t.size(1)
        num_batches = (num_edges + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_edges)
            
            # Get batch edges
            batch_edge_index = edge_index_t[:, start_idx:end_idx]
            batch_edge_attr = edge_attr_t[start_idx:end_idx]
            batch_edge_times = edge_times_t[start_idx:end_idx]
            
            # Get source and destination nodes
            src, dst = batch_edge_index
            
            # Encode edge times
            time_encodings = self.time_encoder(batch_edge_times)
            
            # Combine edge attributes with time encodings
            combined_features = torch.cat([batch_edge_attr, time_encodings], dim=1)
            
            # Update memory for source nodes
            self.memory.update_memory(
                src, 
                embeddings[src], 
                combined_features,
                batch_edge_times
            )
        
        # Get final memory states
        node_memory = self.memory.get_memory()
        
        # Compute node embeddings with graph attention
        # First, construct the input features: node_features + memory + time_encoding
        # We'll use the last timestamp for the global time encoding
        latest_time = edge_times.max()
        global_time_encoding = self.time_encoder(latest_time.repeat(node_features.size(0)))
        
        # Combine node features with memory and time encoding
        augmented_features = torch.cat([node_features, node_memory, global_time_encoding], dim=1)
        
        # Process through embedding layers
        x = augmented_features
        for i, layer in enumerate(self.embedding_layers):
            x = layer(x, edge_index)
            if i < len(self.embedding_layers) - 1:  # Apply non-linearity except at the last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def forward(self, data):
        """
        Forward pass of the TGN model.
        
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
        
        # Compute temporal embeddings
        embeddings = self.compute_temporal_embeddings(
            node_features, edge_index, edge_attr, edge_times
        )
        
        # Apply classifier only to post nodes
        post_embeddings = embeddings[post_mask]
        logits = self.mlp(post_embeddings).squeeze(-1)
        
        return torch.sigmoid(logits)
