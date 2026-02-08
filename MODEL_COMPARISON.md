# TYGR Model Comparison: From RGCN to Hypergraph-RGAT

This document provides a comprehensive comparison of graph neural network architectures for binary type inference in the TYGR project.

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Architectures](#model-architectures)
   - [RGCN (Baseline)](#1-rgcn-relational-graph-convolutional-network---baseline)
   - [GAT](#2-gat-graph-attention-network)
   - [RGAT](#3-rgat-relational-graph-attention-network)
   - [Hypergraph-RGAT](#4-hypergraph-rgat-our-best-model)
4. [Results Summary](#results-summary)
5. [Key Insights](#key-insights)
6. [Conclusion](#conclusion)

---

## Overview

TYGR uses graph neural networks to infer variable types from stripped binaries. The system:
1. Extracts symbolic execution graphs from binaries using angr
2. Encodes AST operations as nodes and edges with different relation types
3. Uses GNNs to learn representations and predict types

We developed and compared four architectures, progressively improving performance.

---

## Dataset

| Split      | Functions | Variables |
|------------|-----------|-----------|
| Train      | 6,648     | 31,732    |
| Validation | 831       | 3,984     |
| Test       | 831       | 3,994     |

- **Source**: Binaries compiled with O0 optimization
- **Edge Types**: 24 reduced operations (mem_data, reg_loc, arithmetic ops, etc.)
- **Output Classes**: 75 type categories

---

## Model Architectures

### 1. RGCN (Relational Graph Convolutional Network) - Baseline

**Architecture**: The original TYGR model uses RGCN, which learns separate transformation matrices for each edge type (relation).

```
Message Passing:
    h_v^{l+1} = σ( Σ_{r∈R} Σ_{u∈N_r(v)} (1/c_{v,r}) W_r^l h_u^l + W_0^l h_v^l )

Where:
    - R = set of relation types (edge operations)
    - N_r(v) = neighbors of v connected by relation r
    - W_r = relation-specific weight matrix
    - c_{v,r} = normalization constant
```

**Key Characteristics**:
- Each edge type has its own learnable weight matrix
- No attention mechanism - all neighbors contribute equally within a relation
- Uses basis decomposition to reduce parameters

**Configuration**:
- 8 message passing layers
- 128-dimensional node embeddings
- Autoregressive type decoder

**Training Command**:
```bash
./TYGR train train.pkl validation.pkl -o model.model -m glow
```

---

### 2. GAT (Graph Attention Network)

**Architecture**: GAT introduces attention mechanisms to learn dynamic edge importance.

```
Attention Coefficients:
    e_{ij} = LeakyReLU(a^T [W h_i || W h_j])
    α_{ij} = softmax_j(e_{ij})

Message Passing:
    h_i^{l+1} = σ( Σ_{j∈N(i)} α_{ij} W h_j )
```

**Key Characteristics**:
- Learns to weight neighbor contributions dynamically
- Multi-head attention for richer representations
- **Problem**: Does not distinguish between edge types!

**Configuration**:
- 4 attention heads
- 8 message passing layers
- 128-dimensional embeddings

**Why GAT Underperformed**:
1. Edge types are critical in TYGR (mem_data vs reg_loc carry different semantics)
2. Vanilla GAT treats all edges identically
3. Attention softmax computed over ALL neighbors, diluting relation-specific signals

**Training Command**:
```bash
./TYGR train train.pkl validation.pkl -o model.model -m glow_gat
```

---

### 3. RGAT (Relational Graph Attention Network)

**Architecture**: RGAT combines RGCN's relation-awareness with GAT's attention mechanism.

```
Per-Relation Attention:
    For each relation r:
        α_{ij}^r = softmax(LeakyReLU(a_r^T [W_r h_i || W_r h_j]))

Message Passing:
    h_i^{l+1} = σ( Σ_{r∈R} Σ_{j∈N_r(i)} α_{ij}^r W_r h_j )

Where:
    - W_r = relation-specific transformation (via basis decomposition)
    - a_r = relation-specific attention parameters
```

**Key Characteristics**:
- **Relation-specific transformations**: Each edge type has its own weight matrix
- **Relation-specific attention**: Attention computed separately per relation
- **Memory-efficient**: Processes edges per relation type to avoid huge tensor allocations
- **Basis decomposition**: W_r = Σ_b c_{rb} B_b (reduces parameters)

**Implementation Details**:
```python
# Basis decomposition for memory efficiency
self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
self.coeffs = nn.Parameter(torch.Tensor(num_relations, num_bases))

# Compute relation-specific weights
rel_weights = torch.einsum('rb,bio->rio', self.coeffs, self.basis)

# Process each relation separately (memory efficient)
for r in range(num_relations):
    mask = (edge_type == r)
    edge_index_r = edge_index[:, mask]
    # ... compute attention and aggregate
```

**Configuration**:
- 4 attention heads
- 10 bases for decomposition
- 60 relation types
- 8 message passing layers

**Training Command**:
```bash
./TYGR train train.pkl validation.pkl -o model.model -m glow_rgat
```

---

### 4. Hypergraph-RGAT (Our Best Model)

**Motivation**: In the original graph, nodes associated with the same variable are not directly connected. The model must learn these relationships through multiple message passing steps.

**Key Insight**: Variables are the prediction targets. Why not make variable relationships explicit?

**Hypergraph Design - Variable-Centric Star Expansion**:

```
Original Graph:
    n1 ----mem_data---- n2 ----reg_loc---- n3
    (all belong to variable V, but no direct connection)

Hypergraph (Star Expansion):
    n1 ----var_hyper---- H_V ----var_hyper---- n2
                          |
                    var_hyper
                          |
                         n3

Where H_V is a hyperedge node representing variable V
```

**Conversion Process**:
```python
def convert_to_hypergraph(glow_input, glow_output):
    for var_idx, var in enumerate(vars_list):
        # Create hyperedge node for this variable
        hyperedge_node = ast_graph.alloc_node(HyperedgeNodeLabel(var_idx))

        # Connect all variable nodes to hyperedge (bidirectional)
        for node in var.nodes:
            ast_graph.add_edge(node, hyperedge_node, 'var_hyper')
            ast_graph.add_edge(hyperedge_node, node, 'var_hyper')
```

**Key Advantages**:
1. **Direct variable-level connectivity**: All nodes of a variable are now 1-hop away via the hyperedge
2. **Learnable aggregation**: The `var_hyper` edge type gets its own attention weights
3. **No data regeneration**: Post-processes existing pkl files
4. **Compatible with existing models**: Just adds a new edge type (62 vs 60 relations)

**Architecture**: Uses the same RGAT architecture, but on the hypergraph structure.

**Configuration**:
- 4 attention heads
- 10 bases for decomposition
- 62 relation types (includes `var_hyper`)
- 8 message passing layers

**Conversion Command**:
```bash
./TYGR hypergraph_convert input.pkl output.pkl
```

**Training Command**:
```bash
./TYGR train train_hyper.pkl validation_hyper.pkl -o model.model -m glow_rgat
```

---

## Results Summary

### Test Set Performance

| Model          | Accuracy | Precision | Recall | F1 Score | Relations |
|----------------|----------|-----------|--------|----------|-----------|
| RGCN (best)    | 65.59%   | 66.22%    | 65.59% | 65.90%   | 60        |
| RGCN (last)    | 65.80%   | 66.86%    | 65.80% | 66.33%   | 60        |
| GAT (best)     | 51.08%   | 51.48%    | 51.08% | 51.28%   | N/A       |
| GAT (last)     | 51.86%   | 52.16%    | 51.86% | 52.01%   | N/A       |
| RGAT (best)    | 66.06%   | 66.89%    | 66.06% | 66.48%   | 60        |
| RGAT (last)    | 67.27%   | 68.41%    | 67.27% | 67.84%   | 60        |
| **Hyper-RGAT (best)** | 68.37%   | 69.24%    | 68.37% | 68.80%   | 62        |
| **Hyper-RGAT (last)** | **69.46%** | **70.73%** | **69.46%** | **70.09%** | 62        |

### Validation Set Performance (Peak)

| Model          | Best Epoch | Accuracy | F1 Score |
|----------------|------------|----------|----------|
| RGCN           | 7          | 67.47%   | 67.64%   |
| GAT            | 33         | 56.31%   | 56.48%   |
| RGAT           | 8          | 68.75%   | 69.09%   |
| Hyper-RGAT     | 10         | 68.19%   | 68.58%   |

### Training Time per Epoch

| Model      | Time/Epoch |
|------------|------------|
| RGCN       | ~4.5 min   |
| GAT        | ~1.0 min   |
| RGAT       | ~2.5 min   |
| Hyper-RGAT | ~2.9 min   |

---

## Key Insights

### 1. Edge Types Matter
GAT's poor performance (51% vs 66%) demonstrates that relation-specific processing is crucial. Binary analysis graphs have semantically distinct edge types that must be treated differently.

### 2. Attention + Relations = Better
RGAT slightly outperforms RGCN by learning dynamic edge importance within each relation type. The attention mechanism helps focus on the most relevant neighbors.

### 3. Hypergraph Structure Provides Significant Boost
The hypergraph representation achieved **+3.7% accuracy** over RGCN and **+2.2%** over RGAT:

| Improvement Over | Accuracy Gain | F1 Gain |
|------------------|---------------|---------|
| RGCN             | +3.87%        | +3.76%  |
| GAT              | +17.60%       | +18.08% |
| RGAT             | +2.19%        | +2.25%  |

### 4. Why Hypergraph Works
- **Reduced path length**: Variable nodes are now 2 hops apart instead of potentially many
- **Explicit variable semantics**: The hyperedge node can learn to aggregate variable-level information
- **Better gradient flow**: Shorter paths mean better gradient propagation during training

### 5. Last Model vs Best Model
Interestingly, the "last" model often outperforms the "best" (early-stopped) model on test:
- RGAT last: 67.27% vs best: 66.06%
- Hyper-RGAT last: 69.46% vs best: 68.37%

This suggests the validation set may not perfectly represent the test distribution, or that continued training provides regularization benefits.

---

## Conclusion

Our progression from RGCN to Hypergraph-RGAT demonstrates three key findings:

1. **Relation-awareness is essential**: Models must treat different edge types distinctly (RGCN, RGAT >> GAT)

2. **Attention mechanisms help**: Learning dynamic edge weights provides modest but consistent improvements (RGAT > RGCN)

3. **Graph structure design matters most**: The hypergraph representation with variable-centric star expansion provides the largest performance gain by making variable relationships explicit

**Best Configuration**: Hypergraph-RGAT with 4 attention heads, 10 bases, and 62 relation types achieves **69.46% accuracy** and **70.09% F1 score** on the test set.

---

## Reproducibility

### Environment
- Python 3.12
- PyTorch (nightly with CUDA 13.0+)
- PyTorch Geometric
- angr 9.2.193

### Commands Summary

```bash
# Generate graphs from binaries
./graph_gen.sh --max-size 150 <BIN_DIR> <DATA_DIR> <LOG_FILE>

# Merge datasets
./dataset_combine.sh <DATA_DIR> <MERGE_DIR> 10 <LOG_FILE>

# Split dataset
./TYGR datasplit merged.pkl --train train.pkl --validation val.pkl --test test.pkl

# Convert to hypergraph
./hypergraph_gen.sh <GRAPH_DIR> <HYPERGRAPH_DIR> <LOG_FILE>

# Train models
./TYGR train train.pkl val.pkl -o model -m glow        # RGCN
./TYGR train train.pkl val.pkl -o model -m glow_gat    # GAT
./TYGR train train.pkl val.pkl -o model -m glow_rgat   # RGAT

# Test models
./TYGR test model.best.model test.pkl
```

---

## File Structure

```
src/
├── methods/
│   ├── glow/           # RGCN (baseline)
│   │   ├── model.py
│   │   └── method.py
│   ├── glow_gat/       # GAT
│   │   ├── model_gat.py
│   │   └── method.py
│   └── glow_rgat/      # RGAT
│       ├── model_rgat.py
│       └── method.py
├── hypergraph_convert.py  # Graph to hypergraph conversion
└── analysis/angr/
    └── edge.py         # Edge type definitions (includes var_hyper)
```
