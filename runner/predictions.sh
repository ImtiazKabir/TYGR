#!/usr/bin/env bash

runner/compare_source.sh predictions/test.c models/rgcn_on_base_graph/rgcn_on_base_graph.best.model -m glow 2>&1 | tee predictions/rgcn_on_base_graph_best.txt
runner/compare_source.sh predictions/test.c models/rgcn_on_base_graph/rgcn_on_base_graph.last.model -m glow 2>&1 | tee predictions/rgcn_on_base_graph_last.txt

runner/compare_source.sh predictions/test.c models/gat_on_base_graph/gat_on_base_graph.best.model -m glow_gat 2>&1 | tee predictions/gat_on_base_graph_best.txt
runner/compare_source.sh predictions/test.c models/gat_on_base_graph/gat_on_base_graph.last.model -m glow_gat 2>&1 | tee predictions/gat_on_base_graph_last.txt

runner/compare_source.sh predictions/test.c models/rgat_on_base_graph/rgat_on_base_graph.best.model -m glow_rgat 2>&1 | tee predictions/rgat_on_base_graph_best.txt
runner/compare_source.sh predictions/test.c models/rgat_on_base_graph/rgat_on_base_graph.last.model -m glow_rgat 2>&1 | tee predictions/rgat_on_base_graph_last.txt

runner/compare_source.sh predictions/test.c models/rgcn_on_star_hypergraph/rgcn_on_star_hypergraph.best.model -m glow 2>&1 | tee predictions/rgcn_on_star_hypergraph_best.txt
runner/compare_source.sh predictions/test.c models/rgcn_on_star_hypergraph/rgcn_on_star_hypergraph.last.model -m glow 2>&1 | tee predictions/rgcn_on_star_hypergraph_last.txt

runner/compare_source.sh predictions/test.c models/rgat_on_star_hypergraph/rgat_on_star_hypergraph.best.model -m glow_rgat 2>&1 | tee predictions/rgat_on_star_hypergraph_best.txt
runner/compare_source.sh predictions/test.c models/rgat_on_star_hypergraph/rgat_on_star_hypergraph.last.model -m glow_rgat 2>&1 | tee predictions/rgat_on_star_hypergraph_last.txt

runner/compare_source.sh predictions/test.c models/rgat_on_true_hypergraph/rgat_on_true_hypergraph.best.model -m glow_hgnn 2>&1 | tee predictions/rgat_on_true_hypergraph_best.txt
runner/compare_source.sh predictions/test.c models/rgat_on_true_hypergraph/rgat_on_true_hypergraph.last.model -m glow_hgnn 2>&1 | tee predictions/rgat_on_true_hypergraph_last.txt

