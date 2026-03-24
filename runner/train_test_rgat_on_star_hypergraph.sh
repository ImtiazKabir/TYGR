#!/usr/bin/env bash

./TYGR train dataset/star_hypergraph_dataset/train.pkl dataset/star_hypergraph_dataset/valid.pkl -o models/rgat_on_star_hypergraph/rgat_on_star_hypergraph -m glow_rgat 2>&1 | tee models/rgat_on_star_hypergraph/train_log.txt
./TYGR test models/rgat_on_star_hypergraph/rgat_on_star_hypergraph.best.model dataset/star_hypergraph_dataset/test.pkl -m glow_rgat 2>&1 | tee models/rgat_on_star_hypergraph/best_test_log.txt
./TYGR test models/rgat_on_star_hypergraph/rgat_on_star_hypergraph.best.model dataset/star_hypergraph_dataset/test.pkl -m glow_rgat 2>&1 | tee models/rgat_on_star_hypergraph/best_test_log.txt
