"""
True Hypergraph Conversion Module

Converts regular graph pkl files to true hypergraph format.
Unlike star expansion (which adds hub nodes), this creates a proper
hyperedge_index tensor for use with HypergraphConv.

The output contains:
- Original graph structure (nodes, edges, edge_types)
- hyperedge_index: [2, num_connections] where row 0 is node indices, row 1 is hyperedge indices
- num_hyperedges: total number of hyperedges (one per variable)

Usage:
    ./TYGR true_hypergraph_convert input.pkl output.pkl

This enables true hypergraph learning while reusing existing graph data.
"""

from argparse import ArgumentParser
import pickle
from typing import List, Tuple, Optional
from tqdm import tqdm
import copy

from .methods.glow.common import GlowInput, GlowOutput, GlowVar
from .analysis.angr.ast_graph import AstGraph, NodeLabel


class HypergraphGlowInput:
    """
    Extended GlowInput that includes hypergraph information.

    Stores the original graph plus hyperedge connectivity.
    """
    def __init__(
        self,
        glow_input: GlowInput,
        hyperedge_index: List[Tuple[int, int]],  # [(node_idx, hyperedge_idx), ...]
        num_hyperedges: int,
    ):
        # Copy all original GlowInput attributes
        self.input_file_name = glow_input.input_file_name
        self.directory = glow_input.directory
        self.file_name = glow_input.file_name
        self.function_name = glow_input.function_name
        self.low_high_pc = glow_input.low_high_pc
        self.ast_graph = glow_input.ast_graph
        self.vars = glow_input.vars
        self.arch = glow_input.arch

        # Hypergraph-specific attributes
        self.hyperedge_index = hyperedge_index  # List of (node_idx, hyperedge_idx)
        self.num_hyperedges = num_hyperedges


def convert_to_true_hypergraph(
    glow_input: GlowInput,
    glow_output: GlowOutput
) -> Tuple[HypergraphGlowInput, GlowOutput]:
    """
    Convert a regular graph to true hypergraph format.

    For each variable V with nodes {n1, n2, ...}:
    - Create a hyperedge H_V (represented by index, not a node)
    - Add entries (n1, H_V), (n2, H_V), ... to hyperedge_index

    The hyperedge_index can be used directly with PyTorch Geometric's HypergraphConv.

    Args:
        glow_input: Original GlowInput with regular graph
        glow_output: Original GlowOutput (unchanged)

    Returns:
        Tuple of (HypergraphGlowInput, original GlowOutput)
    """
    vars_list = glow_input.vars

    hyperedge_index = []  # List of (node_idx, hyperedge_idx)

    for var_idx, var in enumerate(vars_list):
        if len(var.nodes) == 0:
            # No nodes for this variable, still count the hyperedge but no connections
            continue

        # Each variable becomes a hyperedge
        hyperedge_idx = var_idx

        # Connect all nodes of this variable to the hyperedge
        for node in var.nodes:
            hyperedge_index.append((node, hyperedge_idx))

    num_hyperedges = len(vars_list)

    # Create HypergraphGlowInput
    hyper_input = HypergraphGlowInput(
        glow_input=glow_input,
        hyperedge_index=hyperedge_index,
        num_hyperedges=num_hyperedges,
    )

    return (hyper_input, glow_output)


def setup_parser(parser: ArgumentParser):
    parser.add_argument("input", type=str, help="Input pkl file (regular graph)")
    parser.add_argument("output", type=str, help="Output pkl file (true hypergraph)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")


def main(args):
    print(f"Loading {args.input}...")
    with open(args.input, "rb") as f:
        dataset = pickle.load(f)

    print(f"Loaded {len(dataset)} samples")

    # Convert each sample to true hypergraph format
    converted_dataset = []
    stats = {
        'total_samples': len(dataset),
        'total_vars': 0,
        'total_hyperedge_connections': 0,
        'vars_with_nodes': 0,
        'vars_without_nodes': 0,
    }

    for sample in tqdm(dataset, desc="Converting to true hypergraph"):
        (glow_input, glow_output) = sample

        # Count stats before conversion
        num_vars = len(glow_input.vars)
        vars_with_nodes = sum(1 for v in glow_input.vars if len(v.nodes) > 0)
        vars_without_nodes = num_vars - vars_with_nodes

        # Convert
        hyper_input, hyper_output = convert_to_true_hypergraph(glow_input, glow_output)

        stats['total_vars'] += num_vars
        stats['total_hyperedge_connections'] += len(hyper_input.hyperedge_index)
        stats['vars_with_nodes'] += vars_with_nodes
        stats['vars_without_nodes'] += vars_without_nodes

        converted_dataset.append((hyper_input, hyper_output))

        if args.verbose:
            print(f"  {glow_input.function_name}: {num_vars} vars, "
                  f"{len(hyper_input.hyperedge_index)} hyperedge connections")

    # Save converted dataset
    print(f"\nSaving to {args.output}...")
    with open(args.output, "wb") as f:
        pickle.dump(converted_dataset, f)

    # Print summary
    print("\n=== True Hypergraph Conversion Summary ===")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Total hyperedges (variables): {stats['total_vars']}")
    print(f"  - With nodes: {stats['vars_with_nodes']}")
    print(f"  - Without nodes: {stats['vars_without_nodes']}")
    print(f"Total hyperedge connections: {stats['total_hyperedge_connections']}")
    print(f"Avg connections per sample: {stats['total_hyperedge_connections'] / stats['total_samples']:.1f}")
    print(f"Output saved to: {args.output}")
