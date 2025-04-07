"""
Static Graph Convolutional Network (GCN) baseline for misinformation detection.

This model uses discrete graph snapshots without explicit temporal modeling for
comparison with temporal graph models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


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
        self.snapshot_aggregation = config['snapshot_aggregation']
        self.num_snapshots = config['num_snapshots']
        
        # Create GCN layers
        layers = []
        
        # Input layer
        layers.append(GCNConv(self.node_dim, self.hidden_dims[0]))
        
        # Hidden layers
        for i in range(1, len(self.hidden_dims)):
            layers.append(GCNConv(self.hidden_dims[i-1], self.hidden_dims[i]))
        
        self.conv_layers = nn.ModuleList(layers)
        
        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims[-1] // 2, 1)
        )
        
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
        for i, conv in enumerate(self.conv_layers):
            x_new = conv(x, edge_index)
            if i < len(self.conv_layers) - 1:
                x_new = F.relu(x_new)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Add skip connection if dimensions match
            if x.size(1) == x_new.size(1):
                x = x_new + x
            else:
                x = x_new
        
        return x
    
    def aggregate_snapshots(self, snapshot_embeddings):
        """
        Aggregate embeddings from multiple snapshots.
        
        Args:
            snapshot_embeddings (list): List of node embeddings from each snapshot.
            
        Returns:
            torch.Tensor: Aggregated node embeddings.
        """
        if self.snapshot_aggregation == 'mean':
            # Average over snapshots
            return torch.mean(torch.stack(snapshot_embeddings), dim=0)
        elif self.snapshot_aggregation == 'max':
            # Max pooling over snapshots
            return torch.max(torch.stack(snapshot_embeddings), dim=0)[0]
        elif self.snapshot_aggregation == 'last':
            # Use the last snapshot
            return snapshot_embeddings[-1]
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
        
        for snapshot in snapshots:
            edge_index = snapshot['edge_index']
            snapshot_emb = self.forward_snapshot(node_features, edge_index)
            snapshot_embeddings.append(snapshot_emb)
        
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
        
        # GCN layers for each snapshot
        self.gcn_modules = nn.ModuleList([
            GraphConvolutionalNetwork(config) for _ in range(self.num_snapshots)
        ])
        
        # Temporal aggregation weights (learnable)
        self.temporal_weights = nn.Parameter(torch.ones(self.num_snapshots) / self.num_snapshots)
        
        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims[-1] // 2, 1)
        )
        
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
            with torch.no_grad():  # Avoid computing gradients for internal steps
                snapshot_emb = self.gcn_modules[i].forward_snapshot(
                    node_features, snapshot['edge_index']
                )
            
            snapshot_embeddings.append(snapshot_emb)
        
        # Stack embeddings
        stacked_embeddings = torch.stack(snapshot_embeddings)
        
        # Normalize temporal weights
        norm_weights = F.softmax(self.temporal_weights, dim=0)
        
        # Apply weighted combination across temporal snapshots
        combined_embeddings = torch.sum(stacked_embeddings * norm_weights.view(-1, 1, 1), dim=0)
        
        # Apply classifier to post nodes only
        post_embeddings = combined_embeddings[post_mask]
        logits = self.classifier(post_embeddings).squeeze(-1)
        
        return torch.sigmoid(logits)
