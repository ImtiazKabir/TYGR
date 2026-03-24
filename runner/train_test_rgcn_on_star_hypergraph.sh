#!/usr/bin/env bash
./TYGR train dataset/star_hypergraph_dataset/train.pkl dataset/star_hypergraph_dataset/valid.pkl -o models/rgcn_on_star_hypergraph/rgcn_on_star_hypergraph 2>&1 | tee models/rgcn_on_star_hypergraph/train_log.txt
./TYGR test models/rgcn_on_star_hypergraph/rgcn_on_star_hypergraph.best.model dataset/star_hypergraph_dataset/test.pkl 2>&1 | tee models/rgcn_on_star_hypergraph/best_test_log.txt
./TYGR test models/rgcn_on_star_hypergraph/rgcn_on_star_hypergraph.last.model dataset/star_hypergraph_dataset/test.pkl 2>&1 | tee models/rgcn_on_star_hypergraph/last_test_log.txt
