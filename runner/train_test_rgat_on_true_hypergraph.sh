#!/usr/bin/env bash


./TYGR train dataset/true_hypergraph_dataset/train.pkl dataset/true_hypergraph_dataset/valid.pkl -o models/rgat_on_true_hypergraph/rgat_on_true_hypergraph -m glow_hgnn --hgnn-layers 2 --fusion-mode concat 2>&1 | tee models/rgat_on_true_hypergraph/train_log.txt
./TYGR test models/rgat_on_true_hypergraph/rgat_on_true_hypergraph.best.model dataset/true_hypergraph_dataset/test.pkl -m glow_hgnn 2>&1 | tee models/rgat_on_true_hypergraph/best_test_log.txt
./TYGR test models/rgat_on_true_hypergraph/rgat_on_true_hypergraph.last.model dataset/true_hypergraph_dataset/test.pkl -m glow_hgnn 2>&1 | tee models/rgat_on_true_hypergraph/last_test_log.txt
