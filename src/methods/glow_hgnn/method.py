"""
GlowHGNN Method: True Hypergraph Neural Network for type inference.

Combines RGAT for AST edges with HypergraphConv for variable hyperedges.
Requires data converted with true_hypergraph_convert.
"""

from typing import *
from argparse import ArgumentParser
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset

from ..glow.method import Glow
from ..glow import preproc
from ..glow.common import GlowOutput
from .model_hgnn import Config as HGNNConfig, GlowHGNN
from ...true_hypergraph_convert import HypergraphGlowInput


def setup_parser(parser: ArgumentParser):
    """Add HGNN-specific command line arguments."""
    # Note: RGAT parameters (--rgat-heads, --rgat-attention-dropout, --rgat-num-bases)
    # are already defined in glow_rgat.setup_parser()

    # Hypergraph-specific parameters
    parser.add_argument("--hgnn-layers", type=int, default=2,
                        help="Number of HypergraphConv layers (default: 2)")
    parser.add_argument("--hgnn-dropout", type=float, default=0.1,
                        help="Dropout for hypergraph layers (default: 0.1)")
    parser.add_argument("--hgnn-attention", action="store_true", default=True,
                        help="Use attention in HypergraphConv (default: True)")
    parser.add_argument("--no-hgnn-attention", action="store_false", dest="hgnn_attention",
                        help="Disable attention in HypergraphConv")

    # Fusion parameters
    parser.add_argument("--fusion-mode", type=str, default="concat",
                        choices=["concat", "add", "gate"],
                        help="How to fuse RGAT and HGNN representations (default: concat)")


class PreprocHypergraphInput:
    """Preprocessed input with hypergraph information."""
    def __init__(
        self,
        node_labels: Tensor,
        edge_labels: Tensor,
        edges: Tensor,
        var_nodes: List[List[int]],
        hyperedge_index: Tensor,  # [2, num_connections]
    ):
        self.node_labels = node_labels
        self.edge_labels = edge_labels
        self.edges = edges
        self.var_nodes = var_nodes
        self.hyperedge_index = hyperedge_index


def preproc_hypergraph_input(
    hyper_input: HypergraphGlowInput,
    config: preproc.Config
) -> PreprocHypergraphInput:
    """
    Preprocess HypergraphGlowInput into tensors.
    """
    ast_graph = hyper_input.ast_graph

    # Preprocess node labels (same as regular preproc)
    arch = hyper_input.arch if hasattr(hyper_input, "arch") else None
    arch_tensor = preproc.encode_arch(arch, config)
    node_labels = []
    for node_id in ast_graph.graph.nodes:
        node_label = preproc.encode_bit_vector(ast_graph.node_to_label[node_id], config)
        node_features = preproc.encode_node_features(ast_graph.node_to_label[node_id], config)
        node_label_enc = torch.cat([arch_tensor, node_label, node_features], dim=0)
        node_labels.append(node_label_enc)
    node_labels = torch.stack(node_labels)

    # Preprocess edges & edge labels (same as regular preproc)
    edge_labels = []
    edge_tensors = []
    for edge in ast_graph.graph.edges:
        (s, d) = edge
        edge_tensors.append(torch.tensor([s, d], dtype=torch.long))
        edge_labels.append(preproc.encode_edge_label(ast_graph.edge_to_labels[edge], False, config))
        edge_tensors.append(torch.tensor([d, s], dtype=torch.long))
        edge_labels.append(preproc.encode_edge_label(ast_graph.edge_to_labels[edge], True, config))
    edge_labels = torch.tensor(edge_labels)
    edges = torch.transpose(torch.stack(edge_tensors), 0, 1)

    # Preprocess var nodes
    var_nodes = [list(v.nodes) for v in hyper_input.vars]
    if config.flatten_vars:
        var_nodes = [[n] for nodes in var_nodes for n in nodes]

    # Preprocess hyperedge_index
    if hyper_input.hyperedge_index:
        he_nodes = [he[0] for he in hyper_input.hyperedge_index]
        he_edges = [he[1] for he in hyper_input.hyperedge_index]
        hyperedge_index = torch.tensor([he_nodes, he_edges], dtype=torch.long)
    else:
        hyperedge_index = torch.empty((2, 0), dtype=torch.long)

    return PreprocHypergraphInput(node_labels, edge_labels, edges, var_nodes, hyperedge_index)


def hypergraph_collate_fn(list_samples):
    """Collate function for hypergraph batches."""
    y_ids_list = []
    y_list = []

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

    for x, y_ids, y in list_samples:
        y_ids_list.append(y_ids)
        y_list.append(y)

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

    y_ids = torch.cat(y_ids_list, dim=0)
    y = torch.cat(y_list, dim=0)

    # Include hyperedge_index in input_data
    input_data = (list_num_vars, node_labels, edge_labels, edges, var_gather, var_scatter,
                  hyperedge_index, y_ids)
    output_data = (y_ids, y)

    return input_data, output_data


class HypergraphDataset(Dataset):
    """Dataset for hypergraph data."""
    def __init__(self, list_samples, method):
        super().__init__()
        self.list_samples = list_samples
        self.method = method

    def __len__(self):
        return len(self.list_samples)

    def __getitem__(self, idx):
        hyper_input, output = self.list_samples[idx]
        x = preproc_hypergraph_input(hyper_input, self.method.preproc_config)
        y_ids = self.method.preproc_output(hyper_input, output, 'idx')
        y = self.method.preproc_output(hyper_input, output, 'onehot')
        y_ids = y_ids.to(y.device)
        return x, y_ids, y


class GlowHGNNMethod(Glow):
    """
    GlowHGNN: True Hypergraph Neural Network for type inference.

    Uses HypergraphConv for variable hyperedges + RGAT for AST edges.
    """

    def __init__(self, args, phase=None):
        # Call parent init which sets up preproc_config etc.
        super().__init__(args, phase)

        # RGAT parameters
        self.rgat_heads = getattr(args, 'rgat_heads', 4)
        self.rgat_attention_dropout = getattr(args, 'rgat_attention_dropout', 0.1)
        self.rgat_num_bases = getattr(args, 'rgat_num_bases', 10)

        # HGNN parameters
        self.hgnn_layers = getattr(args, 'hgnn_layers', 2)
        self.hgnn_dropout = getattr(args, 'hgnn_dropout', 0.1)
        self.hgnn_attention = getattr(args, 'hgnn_attention', True)
        self.fusion_mode = getattr(args, 'fusion_mode', 'concat')

        # Override model config for training and test
        if phase == "train" or phase == "test":
            self.model_config = HGNNConfig(
                self.preproc_config,
                num_attention_heads=self.rgat_heads,
                attention_dropout=self.rgat_attention_dropout,
                num_bases=self.rgat_num_bases,
                num_hgnn_layers=self.hgnn_layers,
                hgnn_dropout=self.hgnn_dropout,
                use_attention=self.hgnn_attention,
                fusion_mode=self.fusion_mode,
                decoder_type=getattr(args, 'glow_decoder_type', 'autoreg'),
                dropout_rate=getattr(args, 'glow_dropout', 0.0),
                beam_size=getattr(args, 'glow_beam_size', 4),
            )

    def model(self) -> nn.Module:
        """Return the HGNN-based model."""
        return GlowHGNN(self.model_config)

    def dataset(self, ds) -> HypergraphDataset:
        """Return hypergraph-aware dataset."""
        return HypergraphDataset(ds, self)

    def collate_function(self):
        """Return hypergraph-aware collate function."""
        return hypergraph_collate_fn

    def filter_ill_formed(self, list_samples):
        """Filter ill-formed samples from hypergraph data."""
        source_num_functions = len(list_samples)
        source_num_vars = 0
        preproc_num_vars = 0

        result = []
        for sample in list_samples:
            (hyper_input, glow_output) = sample

            # Save original vars for mapping before modification
            original_vars = hyper_input.vars

            source_num_vars += len(original_vars)

            if len(original_vars) == 0:
                continue

            # Create mapping from old var_idx to new var_idx BEFORE modifying
            old_to_new = {}
            new_idx = 0
            for i, v in enumerate(original_vars):
                if len(v.nodes) > 0:
                    old_to_new[i] = new_idx
                    new_idx += 1

            new_vars = [v for v in original_vars if len(v.nodes) > 0]
            new_types = [v for (i, v) in enumerate(glow_output.types) if len(original_vars[i].nodes) > 0]

            preproc_num_vars += len(new_vars)

            if len(new_vars) == 0:
                continue

            # Update vars in hyper_input
            hyper_input.vars = new_vars
            glow_output.types = new_types

            # Filter and remap hyperedge_index
            new_hyperedge_index = []
            for node_idx, hyperedge_idx in hyper_input.hyperedge_index:
                if hyperedge_idx in old_to_new:
                    new_hyperedge_index.append((node_idx, old_to_new[hyperedge_idx]))
            hyper_input.hyperedge_index = new_hyperedge_index

            result.append(sample)

        print("Source #functions:", source_num_functions)
        print("Well Formed #functions:", len(result))
        print("Source #vars:", source_num_vars)
        print("Well Formed #vars:", preproc_num_vars)

        return result

    def preproc_output(self, input, output: GlowOutput, fmt: str = 'onehot') -> Tensor:
        """Preprocess output for hypergraph input."""
        if fmt == 'onehot':
            type_tensors = [self.preproc_config.type_set.type_to_tensor(ty) for ty in output.types]
        else:
            type_tensors = [self.preproc_config.type_set.index_of_type(ty) for ty in output.types]

        # Flatten
        if self.preproc_config.flatten_vars:
            type_tensors = [t for (t, v) in zip(type_tensors, input.vars) for _ in range(len(v.nodes))]
        if fmt == 'onehot':
            return torch.stack(type_tensors)
        return torch.LongTensor(type_tensors)
