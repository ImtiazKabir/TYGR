"""
GlowGAT Method: Graph Attention Network variant of Glow.

This method uses the same data processing pipeline as Glow,
but replaces the RGCN-based model with a GAT-based model.
"""

from typing import *
from argparse import ArgumentParser

from torch import nn

from ..glow.method import Glow
from ..glow import preproc
from .model_gat import Config as GATConfig, GlowGAT


def setup_parser(parser: ArgumentParser):
    """Add GAT-specific command line arguments."""
    parser.add_argument("--gat-heads", type=int, default=4,
                        help="Number of attention heads in GAT (default: 4)")
    parser.add_argument("--gat-attention-dropout", type=float, default=0.1,
                        help="Dropout rate for attention weights (default: 0.1)")
    parser.add_argument("--gat-no-edge-features", action="store_true",
                        help="Disable edge type features in attention")


class GlowGATMethod(Glow):
    """
    GlowGAT: Graph Attention Network for type inference.

    Inherits from Glow to reuse data generation, preprocessing, and evaluation.
    Only overrides the model creation to use GAT instead of RGCN.
    """

    def __init__(self, args, phase=None):
        # Initialize parent (Glow) - this sets up datagen_options, preproc_config, etc.
        super().__init__(args, phase)

        # Store GAT-specific args
        self.gat_heads = getattr(args, 'gat_heads', 4)
        self.gat_attention_dropout = getattr(args, 'gat_attention_dropout', 0.1)
        self.gat_use_edge_features = not getattr(args, 'gat_no_edge_features', False)

        # Override model config for training
        if phase == "train":
            self.model_config = GATConfig(
                self.preproc_config,
                num_attention_heads=self.gat_heads,
                attention_dropout=self.gat_attention_dropout,
                use_edge_features=self.gat_use_edge_features,
                decoder_type=args.glow_decoder_type,
                dropout_rate=args.glow_dropout,
                beam_size=args.glow_beam_size,
            )

    def model(self) -> nn.Module:
        """Return the GAT-based model instead of RGCN."""
        return GlowGAT(self.model_config)
