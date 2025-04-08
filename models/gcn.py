"""
Static Graph Convolutional Network (GCN) baseline for misinformation detection.

This model uses discrete graph snapshots without explicit temporal modeling for
comparison with temporal graph models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class GraphConvolutionalNetwork(nn.Module):
    """
    Static Graph Convolutional Network (GCN) for misinformation detection.
    
    This model aggregates information over static snapshots of the graph without
    explicit temporal dynamics.
    """
    
    def __init__(self, config):
        """
        Initialize the GCN model.
        
        Args:
            config (dict): Model configuration.
        """
        super(GraphConvolutionalNetwork, self).__init__()
        
        self.config = config
        
        # Model dimensions
        self.node_dim = config['node_dim']
        self.hidden_dims = config['hidden_dims']
        self.dropout = config['dropout']
        self.snapshot_aggregation = config.get('snapshot_aggregation', 'mean')
        self.num_snapshots = config.get('num_snapshots', 5)
        self.conv_type = config.get('conv_type', 'gcn')
        self.skip_connections = config.get('skip_connections', True)
        
        # Create GNN layers
        layers = []
        
        # Choose the convolution type
        conv_class = self._get_conv_class()
        
        # Input layer
        layers.append(conv_class(self.node_dim, self.hidden_dims[0]))
        
        # Hidden layers
        for i in range(1, len(self.hidden_dims)):
            layers.append(conv_class(self.hidden_dims[i-1], self.hidden_dims[i]))
        
        self.conv_layers = nn.ModuleList(layers)
        
        # Snapshot weighting (optional learnable temporal weights)
        if config.get('learnable_snapshot_weights', False):
            self.snapshot_weights = nn.Parameter(torch.ones(self.num_snapshots) / self.num_snapshots)
        else:
            self.register_buffer('snapshot_weights', None)
        
        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims[-1] // 2, 1)
        )
    
    def _get_conv_class(self):
        """
        Get the convolution class based on the configuration.
        
        Returns:
            class: The convolution class to use.
        """
        if self.conv_type == 'gcn':
            return GCNConv
        elif self.conv_type == 'sage':
            return SAGEConv
        elif self.conv_type == 'gat':
            # Note: GraphSAGE parameters differ from GCN, so wrap it
            class GATConvWrapper(nn.Module):
                def __init__(self, in_channels, out_channels):
                    super().__init__()
                    self.conv = GATConv(in_channels, out_channels, heads=1)
                
                def forward(self, x, edge_index):
                    return self.conv(x, edge_index)
            return GATConvWrapper
        else:
            return GCNConv  # Default
        
    def forward_snapshot(self, node_features, edge_index):
        """
        Forward pass through GCN for a single snapshot.
        
        Args:
            node_features (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Edge connectivity for the snapshot.
            
        Returns:
            torch.Tensor: Node embeddings for this snapshot.
        """
        x = node_features
        
        # Apply GCN layers with skip connections
        intermediate_features = []
        for i, conv in enumerate(self.conv_layers):
            x_new = conv(x, edge_index)
            
            if i < len(self.conv_layers) - 1:
                x_new = F.relu(x_new)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Add skip connection if dimensions match and skip connections are enabled
            if self.skip_connections and x.size(1) == x_new.size(1):
                x = x_new + x
            else:
                x = x_new
                
            intermediate_features.append(x)
        
        return x, intermediate_features
    
    def aggregate_snapshots(self, snapshot_embeddings):
        """
        Aggregate embeddings from multiple snapshots.
        
        Args:
            snapshot_embeddings (list): List of node embeddings from each snapshot.
            
        Returns:
            torch.Tensor: Aggregated node embeddings.
        """
        # Stacked snapshots [num_snapshots, num_nodes, hidden_dim]
        stacked = torch.stack(snapshot_embeddings)
        
        if self.snapshot_aggregation == 'mean':
            # Average over snapshots
            return torch.mean(stacked, dim=0)
        elif self.snapshot_aggregation == 'max':
            # Max pooling over snapshots
            return torch.max(stacked, dim=0)[0]
        elif self.snapshot_aggregation == 'last':
            # Use the last snapshot
            return snapshot_embeddings[-1]
        elif self.snapshot_aggregation == 'weighted':
            # Apply learned or predefined weights
            if self.snapshot_weights is not None:
                # Normalize weights with softmax
                weights = F.softmax(self.snapshot_weights, dim=0)
                # Apply weights [num_snapshots, 1, 1] * [num_snapshots, num_nodes, hidden_dim]
                weighted = stacked * weights.view(-1, 1, 1)
                return torch.sum(weighted, dim=0)
            else:
                # Default to exponentially increasing weights if not learned
                weights = torch.tensor([2**i for i in range(len(snapshot_embeddings))], 
                                      device=stacked.device, dtype=torch.float)
                weights = weights / weights.sum()
                return (stacked * weights.view(-1, 1, 1)).sum(dim=0)
        else:
            raise ValueError(f"Unknown aggregation method: {self.snapshot_aggregation}")
    
    def forward(self, data):
        """
        Forward pass of the GCN model.
        
        Args:
            data (dict): Graph data containing:
                - node_features: Node feature matrix
                - snapshots: List of graph snapshots with edge indices
                - post_mask: Mask for post nodes
                
        Returns:
            torch.Tensor: Misinformation probability scores for post nodes.
        """
        node_features = data['node_features']
        snapshots = data['snapshots']
        post_mask = data['post_mask']
        
        # Process each snapshot
        snapshot_embeddings = []
        all_intermediate_features = []
        
        for snapshot in snapshots:
            edge_index = snapshot['edge_index']
            snapshot_emb, intermediate_features = self.forward_snapshot(node_features, edge_index)
            snapshot_embeddings.append(snapshot_emb)
            all_intermediate_features.append(intermediate_features)
        
        # Aggregate embeddings from all snapshots
        aggregated_embeddings = self.aggregate_snapshots(snapshot_embeddings)
        
        # Apply classifier to post nodes only
        post_embeddings = aggregated_embeddings[post_mask]
        logits = self.classifier(post_embeddings).squeeze(-1)
        
        return torch.sigmoid(logits)


class TemporalGCN(nn.Module):
    """
    An enhanced static GCN that incorporates minimal temporal information
    through discretized snapshot aggregation.
    
    This model serves as a middle ground between fully static GCN and 
    fully temporal models like TGN and TGAT.
    """
    
    def __init__(self, config):
        """
        Initialize the Temporal GCN model.
        
        Args:
            config (dict): Model configuration.
        """
        super(TemporalGCN, self).__init__()
        
        self.config = config
        
        # Model dimensions
        self.node_dim = config['node_dim']
        self.hidden_dims = config['hidden_dims']
        self.dropout = config['dropout']
        self.num_snapshots = config['num_snapshots']
        self.temporal_fusion = config.get('temporal_fusion', 'attention')
        
        # GCN layers for each snapshot
        self.gcn_modules = nn.ModuleList([
            GraphConvolutionalNetwork(config) for _ in range(self.num_snapshots)
        ])
        
        # Temporal aggregation mechanism
        if self.temporal_fusion == 'attention':
            # Attention-based temporal fusion
            hidden_dim = self.hidden_dims[-1]
            self.temporal_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
        else:
            # Simple weighted fusion
            self.temporal_weights = nn.Parameter(torch.ones(self.num_snapshots) / self.num_snapshots)
        
        # Temporal position encoding (optional)
        if config.get('use_temporal_position', False):
            max_snapshots = 20  # Max number we'd ever expect
            self.temporal_position_embedding = nn.Embedding(max_snapshots, self.hidden_dims[-1])
        else:
            self.temporal_position_embedding = None
        
        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims[-1] // 2, 1)
        )
    
    def temporal_attention_fusion(self, snapshot_embeddings):
        """
        Fuse temporal embeddings using attention.
        
        Args:
            snapshot_embeddings (torch.Tensor): Tensor of shape [num_snapshots, num_nodes, hidden_dim]
            
        Returns:
            torch.Tensor: Attention-weighted node embeddings.
        """
        num_snapshots, num_nodes, hidden_dim = snapshot_embeddings.size()
        
        # Add temporal position embeddings if configured
        if self.temporal_position_embedding is not None:
            positions = torch.arange(num_snapshots, device=snapshot_embeddings.device)
            position_embeddings = self.temporal_position_embedding(positions)  # [num_snapshots, hidden_dim]
            snapshot_embeddings = snapshot_embeddings + position_embeddings.unsqueeze(1)
        
        # Reshape for attention computation
        # [num_snapshots, num_nodes, hidden_dim] -> [num_nodes, num_snapshots, hidden_dim]
        embeddings = snapshot_embeddings.transpose(0, 1)
        
        # Compute attention scores
        # [num_nodes, num_snapshots, hidden_dim] -> [num_nodes, num_snapshots, 1]
        attention_scores = self.temporal_attention(embeddings)
        
        # Normalize attention scores
        # [num_nodes, num_snapshots, 1]
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention weights
        # [num_nodes, num_snapshots, hidden_dim] * [num_nodes, num_snapshots, 1]
        weighted_embeddings = embeddings * attention_weights
        
        # Sum over snapshots
        # [num_nodes, num_snapshots, hidden_dim] -> [num_nodes, hidden_dim]
        fused_embeddings = weighted_embeddings.sum(dim=1)
        
        return fused_embeddings
        
    def forward(self, data):
        """
        Forward pass of the Temporal GCN model.
        
        Args:
            data (dict): Graph data containing:
                - node_features: Node feature matrix
                - snapshots: List of graph snapshots with edge indices
                - post_mask: Mask for post nodes
                
        Returns:
            torch.Tensor: Misinformation probability scores for post nodes.
        """
        node_features = data['node_features']
        snapshots = data['snapshots']
        post_mask = data['post_mask']
        
        # Ensure consistent number of snapshots
        snapshots = snapshots[:self.num_snapshots]
        if len(snapshots) < self.num_snapshots:
            # Duplicate the last snapshot if we have fewer than expected
            snapshots = snapshots + [snapshots[-1]] * (self.num_snapshots - len(snapshots))
        
        # Process each snapshot with its dedicated GCN
        snapshot_embeddings = []
        
        for i, snapshot in enumerate(snapshots):
            # Create a reduced data dict for this snapshot
            snapshot_data = {
                'node_features': node_features,
                'snapshots': [snapshot],  # Wrap in list for compatibility
                'post_mask': post_mask
            }
            
            # Get embeddings from GCN (before classification)
            snapshot_emb, _ = self.gcn_modules[i].forward_snapshot(
                node_features, snapshot['edge_index']
            )
            
            snapshot_embeddings.append(snapshot_emb)
        
        # Stack embeddings [num_snapshots, num_nodes, hidden_dim]
        stacked_embeddings = torch.stack(snapshot_embeddings)
        
        # Combine embeddings using the selected fusion mechanism
        if self.temporal_fusion == 'attention':
            combined_embeddings = self.temporal_attention_fusion(stacked_embeddings)
        else:
            # Use weighted temporal fusion
            weights = F.softmax(self.temporal_weights, dim=0)
            combined_embeddings = torch.sum(stacked_embeddings * weights.view(-1, 1, 1), dim=0)
        
        # Apply classifier to post nodes only
        post_embeddings = combined_embeddings[post_mask]
        logits = self.classifier(post_embeddings).squeeze(-1)
        
        return torch.sigmoid(logits)
