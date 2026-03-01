"""
Hybrid Hypergraph Neural Network for type inference.

Architecture:
1. RGAT layers for AST operation edges (relation-aware)
2. HypergraphConv layers for variable hyperedges
3. Fusion of both representations

This allows the model to learn from both:
- Local AST structure (via RGAT)
- Global variable membership (via hypergraph)
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from torch_scatter import scatter_add, scatter_mean
from torch_scatter import scatter_max as orig_smax

from src.fancy_model import AutoregTypeDecoder
from ..glow.model import NodeLabelEncoder, TypeDecoder, get_aggr
from ..glow_rgat.model_rgat import MemoryEfficientRGATConv
from ..glow import preproc


def scatter_max(src, index, dim=-1, out=None, dim_size=None):
    return orig_smax(src, index, dim, out, dim_size)[0]


class Config:
    """Configuration for Hybrid HGNN model."""
    def __init__(
        self,
        preproc_config: preproc.Config,

        # Node label encoder
        node_encoder_num_layers: int = 2,
        node_latent_dim: int = 128,

        # RGAT specific (for AST edges)
        num_msg_pass_layers: int = 8,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        num_bases: int = None,

        # Hypergraph specific
        num_hgnn_layers: int = 2,
        hgnn_dropout: float = 0.1,
        use_attention: bool = True,  # Use HypergraphConv with attention
        fusion_mode: str = 'concat',  # 'concat', 'add', 'gate'

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
        self.num_bases = num_bases if num_bases else min(8, self.num_relations)

        self.num_hgnn_layers = num_hgnn_layers
        self.hgnn_dropout = hgnn_dropout
        self.use_attention = use_attention
        self.fusion_mode = fusion_mode

        self.num_decoder_layers = num_decoder_layers
        self.decoder_type = decoder_type
        self.beam_size = beam_size
        self.dropout_rate = dropout_rate

        self.node_aggregation = node_aggregation
        self.out_dim = self.preproc_config.type_set.num_types()

        print(f"[GlowHGNN] Hybrid HGNN with {num_msg_pass_layers} RGAT + {num_hgnn_layers} HypergraphConv layers")
        print(f"[GlowHGNN] {self.num_attention_heads} attention heads, {self.num_relations} relations")
        print(f"[GlowHGNN] Fusion mode: {fusion_mode}")
        print(f"[GlowHGNN] Output types: {self.out_dim}")


class HypergraphMessagePassing(nn.Module):
    """
    Hypergraph message passing module using HypergraphConv.

    Processes variable hyperedges to aggregate information
    across all nodes belonging to the same variable.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention

        # HypergraphConv layers (without attention to avoid requiring hyperedge_attr)
        # PyG's HypergraphConv with use_attention=True requires hyperedge_attr
        self.hgnn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else out_channels
            self.hgnn_layers.append(
                HypergraphConv(
                    in_channels=in_dim,
                    out_channels=out_channels,
                    use_attention=False,  # Disable attention (requires hyperedge_attr)
                )
            )

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(out_channels) for _ in range(num_layers)
        ])

        # If attention is desired, we add a self-attention layer after hypergraph conv
        if use_attention:
            self.attention_layers = nn.ModuleList([
                nn.MultiheadAttention(out_channels, num_heads=4, dropout=dropout, batch_first=True)
                for _ in range(num_layers)
            ])

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> Tensor:
        """
        Forward pass through hypergraph layers.

        Args:
            x: Node features [num_nodes, in_channels]
            hyperedge_index: [2, num_connections] - (node_idx, hyperedge_idx)

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        for i, (layer, norm) in enumerate(zip(self.hgnn_layers, self.layer_norms)):
            residual = x if x.size(-1) == self.out_channels else None
            x = layer(x, hyperedge_index)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if residual is not None:
                x = norm(x + residual)
            else:
                x = norm(x)
        return x


class GatedFusion(nn.Module):
    """Gated fusion of RGAT and hypergraph representations."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, x_rgat: Tensor, x_hgnn: Tensor) -> Tensor:
        combined = torch.cat([x_rgat, x_hgnn], dim=-1)
        gate = self.gate(combined)
        return gate * x_rgat + (1 - gate) * x_hgnn


class GlowHGNN(nn.Module):
    """
    Hybrid Hypergraph Neural Network for type inference.

    Combines:
    1. RGAT for AST edge message passing (relation-aware attention)
    2. HypergraphConv for variable hyperedge message passing

    The two branches are fused and fed to the decoder.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.activation = nn.GELU()

        # Encoder (shared)
        self.encoder = NodeLabelEncoder(config)

        # RGAT branch for AST edges
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

        # Hypergraph branch for variable hyperedges
        self.hgnn_branch = HypergraphMessagePassing(
            in_channels=config.node_latent_dim,
            out_channels=config.node_latent_dim,
            num_layers=config.num_hgnn_layers,
            dropout=config.hgnn_dropout,
            use_attention=config.use_attention,
        )

        # Fusion
        if config.fusion_mode == 'concat':
            self.fusion_proj = nn.Linear(config.node_latent_dim * 2, config.node_latent_dim)
        elif config.fusion_mode == 'gate':
            self.fusion = GatedFusion(config.node_latent_dim)
        # 'add' doesn't need extra parameters

        # Decoder
        if config.decoder_type == 'independent':
            self.decoder = TypeDecoder(config)
        elif config.decoder_type == 'autoreg':
            self.decoder = AutoregTypeDecoder(config)
        else:
            raise NotImplementedError(f"Unknown decoder type: {config.decoder_type}")

    def forward(self, x) -> Tensor:
        # Unpack input - now includes hyperedge_index
        (num_vars, node_labels, edge_labels, edges, var_gather, var_scatter,
         hyperedge_index, labels) = x

        # Encode node labels
        node_repr = self.encoder(node_labels)

        # RGAT branch: process AST edges
        x_rgat = node_repr
        for layer in self.rgat_layers:
            x_rgat = layer(x_rgat, edges, edge_labels)
            x_rgat = self.activation(x_rgat)
            x_rgat = F.dropout(x_rgat, p=self.config.dropout_rate, training=self.training)

        # Hypergraph branch: process variable hyperedges
        if hyperedge_index is not None and hyperedge_index.size(1) > 0:
            x_hgnn = self.hgnn_branch(node_repr, hyperedge_index)
        else:
            # Fallback if no hyperedges
            x_hgnn = node_repr

        # Fuse representations
        if self.config.fusion_mode == 'concat':
            x_fused = torch.cat([x_rgat, x_hgnn], dim=-1)
            x_fused = self.fusion_proj(x_fused)
            x_fused = self.activation(x_fused)
        elif self.config.fusion_mode == 'add':
            x_fused = x_rgat + x_hgnn
        elif self.config.fusion_mode == 'gate':
            x_fused = self.fusion(x_rgat, x_hgnn)
        else:
            x_fused = x_rgat  # fallback

        # Readout: aggregate nodes for each variable
        node_repr = x_fused[var_gather]
        readout_agg = get_aggr(self.config)
        var_repr = readout_agg(node_repr, var_scatter, dim=0, dim_size=sum(num_vars))

        # Decode types
        return self.decoder(num_vars, var_repr, labels)
