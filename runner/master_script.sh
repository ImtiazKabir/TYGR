#!/usr/bin/env bash
#runner/source_to_binary.sh dataset/sources dataset/binaries
runner/binary_to_base_graph.sh dataset/binaries dataset/base_graphs .progress/binary_to_base_graph

runner/base_graph_to_star_hypergraph.sh dataset/base_graphs dataset/star_hypergraphs ./.progress/base_graph_to_star_hypergraph
runner/base_graph_to_true_hypergraph.sh dataset/base_graphs dataset/true_hypergraphs ./.progress/base_graph_to_true_hypergraph

runner/graphs_to_merged_dataset.sh dataset/base_graphs dataset/base_graph_dataset 10 ./.progress/base_graphs_to_merged_dataset
runner/graphs_to_merged_dataset.sh dataset/star_hypergraphs dataset/star_hypergraph_dataset 10 ./.progress/star_hypergraphs_to_merged_dataset
runner/graphs_to_merged_dataset.sh dataset/true_hypergraphs dataset/true_hypergraph_dataset 10 ./.progress/true_hypergraphs_to_merged_dataset

runner/merged_dataset_to_train_test_valid.sh

runner/train_test_rgcn_on_base_graph.sh
runner/train_test_gat_on_base_graph.sh
runner/train_test_rgat_on_base_graph.sh
runner/train_test_rgcn_on_star_hypergraph.sh
runner/train_test_rgat_on_star_hypergraph.sh
runner/train_test_rgat_on_true_hypergraph.sh


runner/predictions.sh

