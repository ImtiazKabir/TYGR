"""
GlowRGAT: Relational Graph Attention Network for type inference.

Combines RGCN's relation-specific transformations with GAT's attention mechanism.

Usage:
    ./TYGR train ... -m glow_rgat [--rgat-heads 4] [--rgat-num-bases 10]
    ./TYGR test ... -m glow_rgat
"""

from .method import GlowRGATMethod, setup_parser

__all__ = ['GlowRGATMethod', 'setup_parser']
