#!/usr/bin/env bash

./TYGR datasplit dataset/base_graph_dataset/merged.pkl --train dataset/base_graph_dataset/train.pkl --validation dataset/base_graph_dataset/valid.pkl --test dataset/base_graph_dataset/test.pkl
./TYGR datasplit dataset/star_hypergraph_dataset/merged.pkl --train dataset/star_hypergraph_dataset/train.pkl --validation dataset/star_hypergraph_dataset/valid.pkl --test dataset/star_hypergraph_dataset/test.pkl
./TYGR datasplit dataset/true_hypergraph_dataset/merged.pkl --train dataset/true_hypergraph_dataset/train.pkl --validation dataset/true_hypergraph_dataset/valid.pkl --test dataset/true_hypergraph_dataset/test.pkl
