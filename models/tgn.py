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
    
    def __init__(self, memory_dim, node_features_dim, message_dim, device='cpu'):
        """
        Initialize the memory module.
        
        Args:
            memory_dim (int): Dimension of memory vectors.
            node_features_dim (int): Dimension of node features.
            message_dim (int): Dimension of message vectors.
            device (str): Device to use ('cpu' or 'cuda').
        """
        super(MemoryModule, self).__init__()
        
        self.memory_dim = memory_dim
        self.node_features_dim = node_features_dim
        self.message_dim = message_dim
        self.device = device
        
        # Memory updating mechanism (GRU)
        self.gru = nn.GRUCell(
            input_size=message_dim,
            hidden_size=memory_dim
        )
        
        # Message function: combines node features and message content
        self.message_fn = nn.Sequential(
            nn.Linear(node_features_dim + memory_dim + message_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim)
        )
        
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
            edge_features (torch.Tensor): Features of the edges.
            timestamps (torch.Tensor): Timestamps of the updates.
        """
        if self.memory is None:
            num_nodes = max(node_idxs) + 1
            self.reset_state(num_nodes)
        
        # Compute time delta for temporal effects
        prev_update = self.last_update[node_idxs]
        curr_update = timestamps
        delta_t = curr_update - prev_update
        
        # Prepare memory for message computation
        node_memory = self.memory[node_idxs]
        
        # Compute messages using node features, memory, and edge features
        messages = self.message_fn(
            torch.cat([node_features, node_memory, edge_features], dim=1)
        )
        
        # Aggregate messages for each node
        self.messages[node_idxs] = messages
        
        # Update memory using GRU cell
        updated_memory = self.gru(
            messages,
            node_memory
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
    
    def __init__(self, time_dim):
        """
        Initialize time encoder.
        
        Args:
            time_dim (int): Dimension of time encoding.
        """
        super(TimeEncoder, self).__init__()
        
        self.time_dim = time_dim
        # Learnable parameters for time encoding
        self.w = nn.Parameter(torch.ones(time_dim))
        self.b = nn.Parameter(torch.zeros(time_dim))
        
    def forward(self, timestamps):
        """
        Encode timestamps into high-dimensional vectors.
        
        Args:
            timestamps (torch.Tensor): Tensor of timestamps.
            
        Returns:
            torch.Tensor: Encoded timestamps.
        """
        # Use sine function for encoding
        time_encoding = torch.sin(self.w * timestamps.unsqueeze(1) + self.b)
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
    Temporal Graph Network (TGN) model for misinformation detection.
    
    This model combines graph neural networks with temporal dynamics and node memory
    to learn representations that capture the evolution of misinformation.
    """
    
    def __init__(self, config, num_nodes):
        """
        Initialize the TGN model.
        
        Args:
            config (dict): Model configuration.
            num_nodes (int): Number of nodes in the graph.
        """
        super(TemporalGraphNetwork, self).__init__()
        
        self.config = config
        self.num_nodes = num_nodes
        
        # Model dimensions
        self.node_dim = config['node_dim']
        self.edge_dim = config['edge_dim']
        self.time_dim = config['time_dim']
        self.memory_dim = config['memory_dim']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        # Time encoder
        self.time_encoder = TimeEncoder(self.time_dim)
        
        # Memory module
        self.memory = MemoryModule(
            memory_dim=self.memory_dim,
            node_features_dim=self.node_dim,
            message_dim=self.node_dim + self.edge_dim + self.time_dim,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Graph embedding layers
        self.embedding_layers = nn.ModuleList()
        
        # First layer: node features + memory to hidden dimension
        self.embedding_layers.append(
            GraphAttentionLayer(
                in_dim=self.node_dim + self.memory_dim + self.time_dim,
                out_dim=self.node_dim,
                num_heads=4,
                dropout=self.dropout
            )
        )
        
        # Additional embedding layers
        for _ in range(self.num_layers - 1):
            self.embedding_layers.append(
                GraphAttentionLayer(
                    in_dim=self.node_dim,
                    out_dim=self.node_dim,
                    num_heads=4,
                    dropout=self.dropout
                )
            )
        
        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.node_dim // 2, 1)
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
            
            # Update memory for source nodes
            self.memory.update_memory(
                src, 
                embeddings[src], 
                torch.cat([batch_edge_attr, time_encodings], dim=1),
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
        logits = self.classifier(post_embeddings).squeeze(-1)
        
        return torch.sigmoid(logits)
