import torch
import ptens as p
import torch_geometric.utils as pyg_utils

n=5

g0=p.ggraph.random(n,0.5)
edges = pyg_utils.dense_to_sparse(g0.torch())[0].float()
nedges=edges.size(1)
#print(edges)

G=p.ggraph.from_edge_index(edges).cache(0)

A=torch.randn([3*nedges,3])
B=torch.randn([6*nedges,3])

L0=p.batched_subgraphlayer0b.from_vertex_features([0,0,0],A)
print(L0)

L1=p.batched_subgraphlayer0b.from_edge_features([0,0,0],B)
print(L1)

