"""
GlowHGNN: True Hypergraph Neural Network for type inference.

Uses PyTorch Geometric's HypergraphConv for true hypergraph message passing,
combined with RGAT for regular AST edges.
"""

from .method import GlowHGNNMethod, setup_parser

__all__ = ['GlowHGNNMethod', 'setup_parser']
