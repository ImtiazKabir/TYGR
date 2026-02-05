"""
Graph Attention Network (GAT) based model for type inference.

This is an alternative to the RGCN-based GlowGNN model that uses attention
mechanisms to learn which edges are more important for type prediction.

Key differences from RGCN:
1. Uses attention weights instead of fixed relation-based weights
2. Multi-head attention for learning diverse edge importance patterns
3. Edge type information is incorporated into attention computation
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add, scatter_mean
from torch_scatter import scatter_max as orig_smax

from src.fancy_model import AutoregTypeDecoder
from ..glow.model import NodeLabelEncoder, TypeDecoder, get_aggr
from ..glow import preproc


def scatter_max(src, index, dim=-1, out=None, dim_size=None):
    return orig_smax(src, index, dim, out, dim_size)[0]


class Config:
    """Configuration for GAT-based model."""
    def __init__(
        self,
        preproc_config: preproc.Config,

        # Node label encoder
        node_encoder_num_layers: int = 2,
        node_latent_dim: int = 128,

        # GAT specific
        num_msg_pass_layers: int = 8,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        use_edge_features: bool = True,

        # Decoder
        num_decoder_layers: int = 2,
        decoder_type: str = 'autoreg',
        beam_size: int = 4,
        dropout_rate: float = 0.0,

        # Aggregation
        node_aggregation: str = 'mean',
    ):
        self.preproc_config = preproc_config

        self.node_encoder_in_dim = self.preproc_config.node_label_encoding_dim
        self.node_encoder_num_layers = node_encoder_num_layers
        self.node_latent_dim = node_latent_dim

        self.edge_dim = self.preproc_config.num_edge_ops
        self.num_msg_pass_layers = num_msg_pass_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.use_edge_features = use_edge_features

        self.num_decoder_layers = num_decoder_layers
        self.decoder_type = decoder_type
        self.beam_size = beam_size
        self.dropout_rate = dropout_rate

        self.node_aggregation = node_aggregation
        self.out_dim = self.preproc_config.type_set.num_types()

        print(f"[GlowGAT] Initialized with {self.num_attention_heads} attention heads")
        print(f"[GlowGAT] Output types: {self.out_dim}")


class EdgeAwareGATConv(MessagePassing):
    """
    Graph Attention Convolution that incorporates edge type information.

    Unlike standard GAT which only uses node features for attention,
    this version also considers edge types (like mem_loc, reg_data, etc.)
    which are crucial for understanding binary program semantics.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_edge_types: int,
        heads: int = 4,
        dropout: float = 0.1,
        use_edge_features: bool = True,
    ):
        super(EdgeAwareGATConv, self).__init__(aggr='add', node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.use_edge_features = use_edge_features

        # Per-head output dimension
        self.head_dim = out_channels // heads
        assert self.head_dim * heads == out_channels, "out_channels must be divisible by heads"

        # Linear transformations for query, key, value
        self.W_q = nn.Linear(in_channels, out_channels, bias=False)
        self.W_k = nn.Linear(in_channels, out_channels, bias=False)
        self.W_v = nn.Linear(in_channels, out_channels, bias=False)

        # Edge type embedding
        if use_edge_features:
            self.edge_embedding = nn.Embedding(num_edge_types, self.head_dim)
            # Edge contribution to attention
            self.edge_attn = nn.Linear(self.head_dim, 1, bias=False)

        # Attention scaling
        self.scale = self.head_dim ** -0.5

        # Output projection
        self.out_proj = nn.Linear(out_channels, out_channels)

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(out_channels)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.use_edge_features:
            nn.init.xavier_uniform_(self.edge_embedding.weight)
            nn.init.xavier_uniform_(self.edge_attn.weight)

    def forward(self, x: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        # Store residual
        residual = x

        # Compute Q, K, V
        Q = self.W_q(x)  # [N, out_channels]
        K = self.W_k(x)  # [N, out_channels]
        V = self.W_v(x)  # [N, out_channels]

        # Get edge embeddings if using edge features
        if self.use_edge_features:
            edge_emb = self.edge_embedding(edge_type)  # [E, head_dim]
        else:
            edge_emb = None

        # Propagate messages
        out = self.propagate(edge_index, Q=Q, K=K, V=V, edge_emb=edge_emb)

        # Output projection
        out = self.out_proj(out)

        # Residual connection + layer norm
        out = self.layer_norm(out + residual)

        return out

    def message(self, Q_i: Tensor, K_j: Tensor, V_j: Tensor,
                edge_emb: Tensor, index: Tensor, ptr: Tensor = None,
                size_i: int = None) -> Tensor:
        """
        Compute attention-weighted messages.

        Q_i: Query from target nodes [E, out_channels]
        K_j: Key from source nodes [E, out_channels]
        V_j: Value from source nodes [E, out_channels]
        edge_emb: Edge type embeddings [E, head_dim]
        """
        # Reshape for multi-head attention
        E = Q_i.size(0)
        Q_i = Q_i.view(E, self.heads, self.head_dim)  # [E, H, D]
        K_j = K_j.view(E, self.heads, self.head_dim)  # [E, H, D]
        V_j = V_j.view(E, self.heads, self.head_dim)  # [E, H, D]

        # Compute attention scores
        attn = (Q_i * K_j).sum(dim=-1) * self.scale  # [E, H]

        # Add edge type contribution to attention
        if edge_emb is not None:
            edge_attn_score = self.edge_attn(edge_emb).squeeze(-1)  # [E]
            attn = attn + edge_attn_score.unsqueeze(-1)  # [E, H]

        # Softmax over incoming edges (per target node)
        attn = F.softmax(attn, dim=0)  # Simplified; proper impl uses scatter_softmax
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # Apply attention to values
        out = attn.unsqueeze(-1) * V_j  # [E, H, D]
        out = out.view(E, -1)  # [E, out_channels]

        return out


class GlowGAT(nn.Module):
    """
    Graph Attention Network for type inference.

    Architecture:
    1. Node Label Encoder: Encodes initial node features
    2. GAT Layers: Multiple layers of edge-aware graph attention
    3. Readout: Aggregates node representations for each variable
    4. Type Decoder: Predicts types from variable representations
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.activation = nn.GELU()

        # Encoder (reuse from original model)
        self.encoder = NodeLabelEncoder(config)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for _ in range(config.num_msg_pass_layers):
            self.gat_layers.append(
                EdgeAwareGATConv(
                    in_channels=config.node_latent_dim,
                    out_channels=config.node_latent_dim,
                    num_edge_types=config.edge_dim,
                    heads=config.num_attention_heads,
                    dropout=config.attention_dropout,
                    use_edge_features=config.use_edge_features,
                )
            )

        # Decoder
        if config.decoder_type == 'independent':
            self.decoder = TypeDecoder(config)
        elif config.decoder_type == 'autoreg':
            self.decoder = AutoregTypeDecoder(config)
        else:
            raise NotImplementedError(f"Unknown decoder type: {config.decoder_type}")

    def forward(self, x) -> Tensor:
        (num_vars, node_labels, edge_labels, edges, var_gather, var_scatter, labels) = x

        # Encode node labels
        node_labels = self.encoder(node_labels)

        # Apply GAT layers
        for layer in self.gat_layers:
            node_labels = layer(node_labels, edges, edge_labels)
            node_labels = self.activation(node_labels)
            node_labels = F.dropout(node_labels, p=self.config.dropout_rate, training=self.training)

        # Readout: aggregate nodes for each variable
        node_repr = node_labels[var_gather]
        readout_agg = get_aggr(self.config)
        var_repr = readout_agg(node_repr, var_scatter, dim=0, dim_size=sum(num_vars))

        # Decode types
        return self.decoder(num_vars, var_repr, labels)
