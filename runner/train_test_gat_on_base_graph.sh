#!/usr/bin/env bash

./TYGR train dataset/base_graph_dataset/train.pkl dataset/base_graph_dataset/valid.pkl -o models/gat_on_base_graph/gat_on_base_graph -m glow_gat 2>&1 | tee models/gat_on_base_graph/train_log.txt
./TYGR test models/gat_on_base_graph/gat_on_base_graph.best.model dataset/base_graph_dataset/test.pkl -m glow_gat 2>&1 | tee models/gat_on_base_graph/best_test_log.txt
./TYGR test models/gat_on_base_graph/gat_on_base_graph.last.model dataset/base_graph_dataset/test.pkl -m glow_gat 2>&1 | tee models/gat_on_base_graph/last_test_log.txt

