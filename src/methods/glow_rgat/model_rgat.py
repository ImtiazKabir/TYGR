"""
Relational Graph Attention Network (RGAT) for type inference.

RGAT combines the best of RGCN and GAT:
- Like RGCN: Each edge type has its own transformation weights
- Like GAT: Attention mechanism learns edge importance dynamically

Memory-efficient implementation: processes edges per relation type
to avoid materializing huge tensors.
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as scatter_softmax
from torch_scatter import scatter_add, scatter_mean
from torch_scatter import scatter_max as orig_smax

from src.fancy_model import AutoregTypeDecoder
from ..glow.model import NodeLabelEncoder, TypeDecoder, get_aggr
from ..glow import preproc


def scatter_max(src, index, dim=-1, out=None, dim_size=None):
    return orig_smax(src, index, dim, out, dim_size)[0]


class Config:
    """Configuration for RGAT model."""
    def __init__(
        self,
        preproc_config: preproc.Config,

        # Node label encoder
        node_encoder_num_layers: int = 2,
        node_latent_dim: int = 128,

        # RGAT specific
        num_msg_pass_layers: int = 8,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        num_bases: int = None,  # For basis decomposition (memory efficiency)

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

        self.num_relations = self.preproc_config.num_edge_ops
        self.num_msg_pass_layers = num_msg_pass_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        # Use fewer bases for memory efficiency
        self.num_bases = num_bases if num_bases else min(8, self.num_relations)

        self.num_decoder_layers = num_decoder_layers
        self.decoder_type = decoder_type
        self.beam_size = beam_size
        self.dropout_rate = dropout_rate

        self.node_aggregation = node_aggregation
        self.out_dim = self.preproc_config.type_set.num_types()

        print(f"[GlowRGAT] Initialized with {self.num_attention_heads} attention heads, {self.num_relations} relations")
        print(f"[GlowRGAT] Using {self.num_bases} bases for decomposition")
        print(f"[GlowRGAT] Output types: {self.out_dim}")


class MemoryEfficientRGATConv(nn.Module):
    """
    Memory-efficient Relational Graph Attention Convolution.

    Processes edges per relation type to avoid huge tensor allocations.
    Uses basis decomposition for parameter efficiency.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: int = 8,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = min(num_bases, num_relations)
        self.heads = heads
        self.dropout = dropout

        # Per-head dimension
        self.head_dim = out_channels // heads
        assert self.head_dim * heads == out_channels, "out_channels must be divisible by heads"

        # Basis decomposition for message transformation
        # Instead of R separate [in, out] matrices, use B basis matrices + R coefficient vectors
        self.basis = nn.Parameter(torch.Tensor(self.num_bases, in_channels, out_channels))
        self.coeffs = nn.Parameter(torch.Tensor(num_relations, self.num_bases))

        # Attention parameters per relation (also with basis decomposition)
        # att_l and att_r for source and target node contributions
        self.att_src = nn.Parameter(torch.Tensor(num_relations, heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(num_relations, heads, self.head_dim))

        # Bias and layer norm
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.layer_norm = nn.LayerNorm(out_channels)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.basis)
        nn.init.xavier_uniform_(self.coeffs)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        """
        Memory-efficient forward pass.

        Processes each relation type separately to avoid huge tensor allocations.
        """
        num_nodes = x.size(0)
        residual = x

        # Compute relation-specific weight matrices via basis decomposition
        # Shape: [num_relations, in_channels, out_channels]
        # We compute this once and reuse
        rel_weights = torch.einsum('rb,bio->rio', self.coeffs, self.basis)

        # Initialize output
        out = torch.zeros(num_nodes, self.out_channels, device=x.device, dtype=x.dtype)

        # Process each relation type separately (memory efficient)
        for r in range(self.num_relations):
            # Get edges of this relation type
            mask = (edge_type == r)
            if not mask.any():
                continue

            # Get edge indices for this relation
            edge_index_r = edge_index[:, mask]
            src_idx = edge_index_r[0]  # Source nodes
            dst_idx = edge_index_r[1]  # Target nodes

            # Get source and target node features
            x_src = x[src_idx]  # [E_r, in_channels]
            x_dst = x[dst_idx]  # [E_r, in_channels]

            # Transform source nodes with relation-specific weights
            # [E_r, in_channels] @ [in_channels, out_channels] -> [E_r, out_channels]
            msg = torch.matmul(x_src, rel_weights[r])

            # Compute attention scores
            # Reshape for multi-head: [E_r, heads, head_dim]
            msg_heads = msg.view(-1, self.heads, self.head_dim)
            x_dst_transformed = torch.matmul(x_dst, rel_weights[r])
            x_dst_heads = x_dst_transformed.view(-1, self.heads, self.head_dim)

            # Attention: dot product with relation-specific attention vectors
            # alpha = (x_dst * att_dst + msg * att_src).sum(-1)
            alpha = (x_dst_heads * self.att_dst[r]).sum(-1) + (msg_heads * self.att_src[r]).sum(-1)
            alpha = F.leaky_relu(alpha, negative_slope=0.2)

            # Softmax over neighbors (per target node, within this relation)
            alpha = scatter_softmax(alpha, dst_idx, dim=0)  # [E_r, heads]
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

            # Apply attention and aggregate
            weighted_msg = msg_heads * alpha.unsqueeze(-1)  # [E_r, heads, head_dim]
            weighted_msg = weighted_msg.view(-1, self.out_channels)  # [E_r, out_channels]

            # Scatter add to target nodes
            out = out.scatter_add(0, dst_idx.unsqueeze(-1).expand_as(weighted_msg), weighted_msg)

        out = out + self.bias

        # Residual connection + layer norm
        if residual.size(-1) == out.size(-1):
            out = self.layer_norm(out + residual)
        else:
            out = self.layer_norm(out)

        return out


class GlowRGAT(nn.Module):
    """
    Relational Graph Attention Network for type inference.

    Architecture:
    1. Node Label Encoder: Encodes initial node features
    2. RGAT Layers: Relation-aware attention message passing
    3. Readout: Aggregates node representations for each variable
    4. Type Decoder: Predicts types from variable representations
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.activation = nn.GELU()

        # Encoder (reuse from original model)
        self.encoder = NodeLabelEncoder(config)

        # RGAT layers
        self.rgat_layers = nn.ModuleList()
        for i in range(config.num_msg_pass_layers):
            self.rgat_layers.append(
                MemoryEfficientRGATConv(
                    in_channels=config.node_latent_dim,
                    out_channels=config.node_latent_dim,
                    num_relations=config.num_relations,
                    num_bases=config.num_bases,
                    heads=config.num_attention_heads,
                    dropout=config.attention_dropout,
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

        # Apply RGAT layers
        for layer in self.rgat_layers:
            node_labels = layer(node_labels, edges, edge_labels)
            node_labels = self.activation(node_labels)
            node_labels = F.dropout(node_labels, p=self.config.dropout_rate, training=self.training)

        # Readout: aggregate nodes for each variable
        node_repr = node_labels[var_gather]
        readout_agg = get_aggr(self.config)
        var_repr = readout_agg(node_repr, var_scatter, dim=0, dim_size=sum(num_vars))

        # Decode types
        return self.decoder(num_vars, var_repr, labels)
