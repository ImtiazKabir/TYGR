"""
GlowGAT: Graph Attention Network variant for type inference.

Usage:
    ./TYGR train ... -m glow_gat [--gat-heads 4] [--gat-attention-dropout 0.1]
    ./TYGR test ... -m glow_gat
"""

from .method import GlowGATMethod, setup_parser

__all__ = ['GlowGATMethod', 'setup_parser']
