"""
GlowRGAT Method: Relational Graph Attention Network variant of Glow.

Combines the relation-awareness of RGCN with the attention mechanism of GAT.
Each edge type has its own attention computation via basis decomposition.
"""

from typing import *
from argparse import ArgumentParser

from torch import nn

from ..glow.method import Glow
from ..glow import preproc
from .model_rgat import Config as RGATConfig, GlowRGAT


def setup_parser(parser: ArgumentParser):
    """Add RGAT-specific command line arguments."""
    parser.add_argument("--rgat-heads", type=int, default=4,
                        help="Number of attention heads in RGAT (default: 4)")
    parser.add_argument("--rgat-attention-dropout", type=float, default=0.1,
                        help="Dropout rate for attention weights (default: 0.1)")
    parser.add_argument("--rgat-num-bases", type=int, default=10,
                        help="Number of basis matrices for weight decomposition (default: 10)")


class GlowRGATMethod(Glow):
    """
    GlowRGAT: Relational Graph Attention Network for type inference.

    Inherits from Glow to reuse data generation, preprocessing, and evaluation.
    Only overrides the model creation to use RGAT instead of RGCN.
    """

    def __init__(self, args, phase=None):
        # Initialize parent (Glow) - this sets up datagen_options, preproc_config, etc.
        super().__init__(args, phase)

        # Store RGAT-specific args
        self.rgat_heads = getattr(args, 'rgat_heads', 4)
        self.rgat_attention_dropout = getattr(args, 'rgat_attention_dropout', 0.1)
        self.rgat_num_bases = getattr(args, 'rgat_num_bases', 10)

        # Override model config for training
        if phase == "train":
            self.model_config = RGATConfig(
                self.preproc_config,
                num_attention_heads=self.rgat_heads,
                attention_dropout=self.rgat_attention_dropout,
                num_bases=self.rgat_num_bases,
                decoder_type=args.glow_decoder_type,
                dropout_rate=args.glow_dropout,
                beam_size=args.glow_beam_size,
            )

    def model(self) -> nn.Module:
        """Return the RGAT-based model instead of RGCN."""
        return GlowRGAT(self.model_config)
