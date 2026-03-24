"""
Compare actual types (from DWARF) vs inferred types (from model).

Usage:
    ./TYGR compare_types <model> <binary_with_debug_info> [-m method] [-v]

This extracts ground truth types from DWARF info and compares them
with model predictions, showing variable name, actual type, and inferred type.
"""

from argparse import ArgumentParser
import pickle
from typing import Dict, List, Tuple
from collections import defaultdict

import torch

from .methods import get_method, setup_parser as methods_setup_parser, DEFAULT_METHOD
from .methods.glow.common import GlowInput, GlowOutput
from .methods.glow import datagen, predict, preproc
from .methods.glow.datagen import Options
from .analysis.types.btypes import type_to_btype
from .utils.cuda import setup_cuda
from .true_hypergraph_convert import convert_to_true_hypergraph, HypergraphGlowInput
from .methods.glow_hgnn.method import preproc_hypergraph_input


def collate_hypergraph_inputs(list_x):
    """Collate hypergraph preprocessed inputs for inference (no labels)."""
    node_labels = []
    edges = []
    edge_labels = []
    hyperedge_list = []
    node_offset = 0
    hyperedge_offset = 0
    var_gather = []
    var_scatter = []
    num_vars = 0
    list_num_vars = []

    for x in list_x:
        node_labels.append(x.node_labels)
        cur_edges = x.edges + node_offset
        edges.append(cur_edges)
        edge_labels.append(x.edge_labels)

        # Handle hyperedge_index with offset
        if x.hyperedge_index.size(1) > 0:
            he_idx = x.hyperedge_index.clone()
            he_idx[0] += node_offset  # offset node indices
            he_idx[1] += hyperedge_offset  # offset hyperedge indices
            hyperedge_list.append(he_idx)
            hyperedge_offset += x.hyperedge_index[1].max().item() + 1

        for nodes in x.var_nodes:
            var_gather += [t + node_offset for t in nodes]
            var_scatter += [num_vars] * len(nodes)
            num_vars += 1
        list_num_vars.append(len(x.var_nodes))
        node_offset += x.node_labels.shape[0]

    node_labels = torch.cat(node_labels, dim=0)
    edge_labels = torch.cat(edge_labels, dim=0)
    edges = torch.cat(edges, dim=1)
    var_gather = torch.LongTensor(var_gather).to(node_labels.device)
    var_scatter = torch.LongTensor(var_scatter).to(var_gather.device)

    # Concatenate hyperedge indices
    if hyperedge_list:
        hyperedge_index = torch.cat(hyperedge_list, dim=1)
    else:
        hyperedge_index = torch.empty((2, 0), dtype=torch.long)

    return list_num_vars, node_labels, edge_labels, edges, var_gather, var_scatter, hyperedge_index


def setup_parser(parser: ArgumentParser):
    parser.add_argument("model", type=str, help="Trained model file")
    parser.add_argument("binary", type=str, help="Binary with debug info (-g)")
    parser.add_argument("-m", "--method", type=str, default=DEFAULT_METHOD)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-g", "--gpu", type=int, default=None)
    parser.add_argument("--type-set", type=str, default="rstd")
    parser.add_argument("--parallel", action="store_true", default=True)
    parser.add_argument("--no-splice", action="store_true")
    parser.add_argument("--ignore-functions", type=str, default="src/data/ignore_functions.json")
    parser.add_argument("--output-functions", type=str, default=None)

    methods_setup_parser(parser)


def main(args):
    # GPU setup
    setup_cuda(args.gpu)

    # Load model
    print(f"Loading model: {args.model}")
    model = torch.load(args.model, weights_only=False)
    model.eval()

    # Use the model's saved preprocessing config (NOT a newly created one)
    # This ensures we use the same settings the model was trained with
    preproc_config = model.config.preproc_config

    # Check if using hypergraph method
    is_hypergraph = args.method == "glow_hgnn"

    # Generate dataset WITH ground truth types using datagen
    print(f"Analyzing binary: {args.binary}")
    print(f"Method: {args.method} (hypergraph: {is_hypergraph})")

    datagen_options = Options(
        verbose=args.verbose,
        parallel=args.parallel,
        no_splice=args.no_splice,
        ignore_functions_file=args.ignore_functions,
        predict_phase=False,  # We want ground truth types
    )

    # Generate dataset - this extracts DWARF types as ground truth
    dataset = list(datagen.generate_glow_dataset(args.binary, "/tmp", datagen_options))

    if not dataset:
        print("No functions with variables found in binary!")
        return

    print(f"Found {len(dataset)} functions with variables\n")
    print("=" * 80)

    total_vars = 0
    correct_vars = 0

    for (glow_input, glow_output) in dataset:
        func_name = glow_input.function_name
        func_addr = glow_input.low_high_pc[0]

        if len(glow_input.vars) == 0:
            continue

        # Filter vars with no nodes (same as training)
        valid_indices = [i for i, v in enumerate(glow_input.vars) if len(v.nodes) > 0]
        if not valid_indices:
            continue

        print(f"\nFunction: {func_name} @ {hex(func_addr)}")
        print("-" * 70)

        # Filter to valid vars
        filtered_vars = [glow_input.vars[i] for i in valid_indices]
        filtered_types = [glow_output.types[i] for i in valid_indices]


        print(f"{'Variable':<20} {'Actual Type':<25} {'Inferred Type':<20} {'Match'}")
        print("-" * 70)

        # Create filtered input for preprocessing
        glow_input.vars = filtered_vars
        glow_output.types = filtered_types

        # Preprocess input
        try:
            if is_hypergraph:
                # Convert to hypergraph format
                hyper_input, hyper_output = convert_to_true_hypergraph(glow_input, glow_output)
                x = preproc_hypergraph_input(hyper_input, preproc_config)
                num_vars, node_labels, edge_labels, edges, var_gather, var_scatter, hyperedge_index = collate_hypergraph_inputs([x])
                model_input = (num_vars, node_labels, edge_labels, edges, var_gather, var_scatter, hyperedge_index, None)
            else:
                # Regular glow/gat/rgat preprocessing - use model's saved config
                x = preproc.preproc_input(glow_input, preproc_config)
                num_vars, node_labels, edge_labels, edges, var_gather, var_scatter = predict.collate_inputs([x])
                model_input = (num_vars, node_labels, edge_labels, edges, var_gather, var_scatter, None)

            with torch.no_grad():
                y_pred, _ = model(model_input)

            # Handle flatten_vars: when True, model predicts per-node, not per-variable
            # We need to map predictions back to variables using majority voting
            if preproc_config.flatten_vars:
                # Build mapping: predictions are ordered by flattened nodes
                # For each variable, use majority vote from its nodes
                from collections import Counter
                pred_idx = 0
                var_predictions = []
                for var in filtered_vars:
                    num_nodes = len(var.nodes)
                    if pred_idx + num_nodes <= y_pred.size(0):
                        # Get predictions for all nodes of this variable
                        node_preds = y_pred[pred_idx:pred_idx + num_nodes]
                        # Majority vote: find most common predicted type index
                        pred_indices = [p.argmax().item() for p in node_preds]
                        most_common_idx = Counter(pred_indices).most_common(1)[0][0]
                        # Create a one-hot tensor for the majority vote
                        var_pred = torch.zeros_like(y_pred[0])
                        var_pred[most_common_idx] = 1.0
                        var_predictions.append(var_pred)
                    pred_idx += num_nodes
                if var_predictions:
                    y_pred_per_var = torch.stack(var_predictions)
                else:
                    y_pred_per_var = y_pred
            else:
                y_pred_per_var = y_pred

            # Compare predictions with ground truth
            for i, (var, actual_type) in enumerate(zip(filtered_vars, filtered_types)):
                var_name = var.name if var.name else f"var_{i}"

                # Get actual type (from DWARF via datagen)
                actual_str = str(actual_type)

                # Get inferred type - use model's saved type_set
                if i < y_pred_per_var.size(0):
                    inferred_type = preproc_config.type_set.tensor_to_type(y_pred_per_var[i])
                    inferred_str = str(inferred_type)
                else:
                    inferred_str = "???"

                # Check match
                match = "Y" if actual_str == inferred_str else ""

                if match:
                    correct_vars += 1

                print(f"{var_name:<20} {actual_str:<25} {inferred_str:<20} {match}")
                total_vars += 1

        except Exception as e:
            print(f"  Error processing function: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"Total variables: {total_vars}")
    print(f"Correct predictions: {correct_vars}")
    if total_vars > 0:
        print(f"Accuracy: {correct_vars / total_vars * 100:.2f}%")
