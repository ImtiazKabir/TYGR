#!/usr/bin/env bash
runner/source_to_binary.sh dataset/sources dataset/binaries
runner/binary_to_base_graph.sh dataset/binaries dataset/base_graphs .progress/binary_to_base_graph

runner/base_graph_to_star_hypergraph.sh dataset/base_graphs dataset/star_hypergraphs ./.progress/base_graph_to_star_hypergraph
runner/base_graph_to_true_hypergraph.sh dataset/base_graphs dataset/true_hypergraphs ./.progress/base_graph_to_true_hypergraph

runner/graphs_to_merged_dataset.sh dataset/base_graphs dataset/base_graph_dataset 10 ./.progress/base_graphs_to_merged_dataset
runner/graphs_to_merged_dataset.sh dataset/star_hypergraphs dataset/star_hypergraph_dataset 10 ./.progress/star_hypergraphs_to_merged_dataset
runner/graphs_to_merged_dataset.sh dataset/true_hypergraphs dataset/true_hypergraph_dataset 10 ./.progress/true_hypergraphs_to_merged_dataset

./TYGR datasplit dataset/base_graph_dataset/merged.pkl --train dataset/base_graph_dataset/train.pkl --validation dataset/base_graph_dataset/valid.pkl --test dataset/base_graph_dataset/test.pkl
./TYGR datasplit dataset/star_hypergraph_dataset/merged.pkl --train dataset/star_hypergraph_dataset/train.pkl --validation dataset/star_hypergraph_dataset/valid.pkl --test dataset/star_hypergraph_dataset/test.pkl
./TYGR datasplit dataset/true_hypergraph_dataset/merged.pkl --train dataset/true_hypergraph_dataset/train.pkl --validation dataset/true_hypergraph_dataset/valid.pkl --test dataset/true_hypergraph_dataset/test.pkl

./TYGR train dataset/base_graph_dataset/train.pkl dataset/base_graph_dataset/valid.pkl -o models/rgcn_on_base_graph/rgcn_on_base_graph 2>&1 | tee models/rgcn_on_base_graph/train_log.txt
./TYGR test models/rgcn_on_base_graph/rgcn_on_base_graph.best.model dataset/base_graph_dataset/test.pkl 2>&1 | tee models/rgcn_on_base_graph/best_test_log.txt
./TYGR test models/rgcn_on_base_graph/rgcn_on_base_graph.last.model dataset/base_graph_dataset/test.pkl 2>&1 | tee models/rgcn_on_base_graph/last_test_log.txt

./TYGR train dataset/base_graph_dataset/train.pkl dataset/base_graph_dataset/valid.pkl -o models/gat_on_base_graph/gat_on_base_graph -m glow_gat 2>&1 | tee models/gat_on_base_graph/train_log.txt
./TYGR test models/gat_on_base_graph/gat_on_base_graph.best.model dataset/base_graph_dataset/test.pkl -m glow_gat 2>&1 | tee models/gat_on_base_graph/best_test_log.txt
./TYGR test models/gat_on_base_graph/gat_on_base_graph.last.model dataset/base_graph_dataset/test.pkl -m glow_gat 2>&1 | tee models/gat_on_base_graph/last_test_log.txt

./TYGR train dataset/base_graph_dataset/train.pkl dataset/base_graph_dataset/valid.pkl -o models/rgat_on_base_graph/rgat_on_base_graph -m glow_rgat 2>&1 | tee models/rgat_on_base_graph/train_log.txt
./TYGR test models/rgat_on_base_graph/rgat_on_base_graph.best.model dataset/base_graph_dataset/test.pkl -m glow_rgat 2>&1 | tee models/rgat_on_base_graph/best_test_log.txt
./TYGR test models/rgat_on_base_graph/rgat_on_base_graph.last.model dataset/base_graph_dataset/test.pkl -m glow_rgat 2>&1 | tee models/rgat_on_base_graph/last_test_log.txt

./TYGR train dataset/star_hypergraph_dataset/train.pkl dataset/star_hypergraph_dataset/valid.pkl -o models/rgcn_on_star_hypergraph/rgcn_on_star_hypergraph 2>&1 | tee models/rgcn_on_star_hypergraph/train_log.txt
./TYGR test models/rgcn_on_star_hypergraph/rgcn_on_star_hypergraph.best.model dataset/star_hypergraph_dataset/test.pkl 2>&1 | tee models/rgcn_on_star_hypergraph/best_test_log.txt
./TYGR test models/rgcn_on_star_hypergraph/rgcn_on_star_hypergraph.last.model dataset/star_hypergraph_dataset/test.pkl 2>&1 | tee models/rgcn_on_star_hypergraph/last_test_log.txt

./TYGR train dataset/star_hypergraph_dataset/train.pkl dataset/star_hypergraph_dataset/valid.pkl -o models/rgat_on_star_hypergraph/rgat_on_star_hypergraph -m glow_rgat 2>&1 | tee models/rgat_on_star_hypergraph/train_log.txt
./TYGR test models/rgat_on_star_hypergraph/rgat_on_star_hypergraph.best.model dataset/star_hypergraph_dataset/test.pkl -m glow_rgat 2>&1 | tee models/rgat_on_star_hypergraph/best_test_log.txt
./TYGR test models/rgat_on_star_hypergraph/rgat_on_star_hypergraph.best.model dataset/star_hypergraph_dataset/test.pkl -m glow_rgat 2>&1 | tee models/rgat_on_star_hypergraph/best_test_log.txt

./TYGR train dataset/true_hypergraph_dataset/train.pkl dataset/true_hypergraph_dataset/valid.pkl -o models/rgat_on_true_hypergraph/rgat_on_true_hypergraph -m glow_hgnn --hgnn-layers 2 --fusion-mode concat 2>&1 | tee models/rgat_on_true_hypergraph/train_log.txt
./TYGR test models/rgat_on_true_hypergraph/rgat_on_true_hypergraph.best.model dataset/true_hypergraph_dataset/test.pkl -m glow_hgnn 2>&1 | tee models/rgat_on_true_hypergraph/best_test_log.txt
./TYGR test models/rgat_on_true_hypergraph/rgat_on_true_hypergraph.last.model dataset/true_hypergraph_dataset/test.pkl -m glow_hgnn 2>&1 | tee models/rgat_on_true_hypergraph/last_test_log.txt
