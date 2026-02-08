"""
Hypergraph Conversion Module

Converts regular graph pkl files to hypergraph pkl files using star expansion.
For each variable, creates a hyperedge node that connects all nodes associated
with that variable.

Usage:
    ./TYGR hypergraph_convert input.pkl output.pkl

This enables hypergraph-based learning while reusing existing graph data.
"""

from argparse import ArgumentParser
import pickle
from typing import List, Tuple
from tqdm import tqdm
import copy

from .methods.glow.common import GlowInput, GlowOutput, GlowVar
from .analysis.angr.ast_graph import AstGraph, NodeLabel


class HyperedgeNodeLabel:
    """
    Special node label for hyperedge nodes.
    These nodes represent variable-centric hyperedges in the star expansion.
    """
    def __init__(self, var_index: int, var_name: str = None):
        self.var_index = var_index
        self.var_name = var_name
        self.addr = None  # Hyperedge nodes don't have addresses
        self.ast_val = 0  # Placeholder value
        self.bitsize = 64  # Default bitsize for hyperedge nodes

    def __str__(self):
        return f"HyperedgeNode(var={self.var_index}, name={self.var_name})"


def convert_to_hypergraph(glow_input: GlowInput, glow_output: GlowOutput) -> Tuple[GlowInput, GlowOutput]:
    """
    Convert a regular graph to a hypergraph using star expansion.

    For each variable V with nodes {n1, n2, ...}:
    1. Create a hyperedge node H_V
    2. Add edges from each n_i to H_V with edge type 'var_hyper'
    3. Add the hyperedge node to the variable's node list

    Args:
        glow_input: Original GlowInput with regular graph
        glow_output: Original GlowOutput (unchanged)

    Returns:
        Tuple of (modified GlowInput, original GlowOutput)
    """
    ast_graph = glow_input.ast_graph
    vars_list = glow_input.vars

    # Track new nodes and edges to add
    new_vars = []

    for var_idx, var in enumerate(vars_list):
        if len(var.nodes) == 0:
            # No nodes for this variable, skip
            new_vars.append(var)
            continue

        # Create hyperedge node for this variable
        hyperedge_label = HyperedgeNodeLabel(var_idx, var.name)
        hyperedge_node = ast_graph.alloc_node(hyperedge_label)

        # Add edges from each variable node to the hyperedge node
        for node in var.nodes:
            # Edge from variable node to hyperedge node
            ast_graph.add_edge(node, hyperedge_node, 'var_hyper')
            # Also add reverse edge for bidirectional message passing
            ast_graph.add_edge(hyperedge_node, node, 'var_hyper')

        # Create new GlowVar with hyperedge node added to its nodes
        new_nodes = list(var.nodes) + [hyperedge_node]
        new_var = GlowVar(var.name, var.locs, new_nodes)
        new_vars.append(new_var)

    # Create new GlowInput with modified graph and vars
    new_glow_input = GlowInput(
        input_file_name=glow_input.input_file_name,
        directory=glow_input.directory,
        file_name=glow_input.file_name,
        function_name=glow_input.function_name,
        low_high_pc=glow_input.low_high_pc,
        ast_graph=ast_graph,  # Modified in place
        vars=new_vars,
        arch=glow_input.arch,
    )

    return (new_glow_input, glow_output)


def setup_parser(parser: ArgumentParser):
    parser.add_argument("input", type=str, help="Input pkl file (regular graph)")
    parser.add_argument("output", type=str, help="Output pkl file (hypergraph)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")


def main(args):
    print(f"Loading {args.input}...")
    with open(args.input, "rb") as f:
        dataset = pickle.load(f)

    print(f"Loaded {len(dataset)} samples")

    # Convert each sample to hypergraph
    converted_dataset = []
    stats = {
        'total_samples': len(dataset),
        'total_vars': 0,
        'total_hyperedge_nodes': 0,
        'total_hyperedge_edges': 0,
    }

    for sample in tqdm(dataset, desc="Converting to hypergraph"):
        (glow_input, glow_output) = sample

        # Count stats before conversion
        num_vars = len(glow_input.vars)
        num_nodes_before = len(glow_input.ast_graph.graph.nodes)
        num_edges_before = len(glow_input.ast_graph.graph.edges)

        # Convert
        new_input, new_output = convert_to_hypergraph(glow_input, glow_output)

        # Count stats after conversion
        num_nodes_after = len(new_input.ast_graph.graph.nodes)
        num_edges_after = len(new_input.ast_graph.graph.edges)

        stats['total_vars'] += num_vars
        stats['total_hyperedge_nodes'] += (num_nodes_after - num_nodes_before)
        stats['total_hyperedge_edges'] += (num_edges_after - num_edges_before)

        converted_dataset.append((new_input, new_output))

        if args.verbose:
            print(f"  {glow_input.function_name}: {num_vars} vars, "
                  f"+{num_nodes_after - num_nodes_before} nodes, "
                  f"+{num_edges_after - num_edges_before} edges")

    # Save converted dataset
    print(f"\nSaving to {args.output}...")
    with open(args.output, "wb") as f:
        pickle.dump(converted_dataset, f)

    # Print summary
    print("\n=== Conversion Summary ===")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Total variables: {stats['total_vars']}")
    print(f"Hyperedge nodes added: {stats['total_hyperedge_nodes']}")
    print(f"Hyperedge edges added: {stats['total_hyperedge_edges']}")
    print(f"Output saved to: {args.output}")
