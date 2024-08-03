import torch
import ptens_base
import ptens

E=ptens.subgraph.edge()
print(E)
T=ptens.subgraph.triangle()
print(T)
C=ptens.subgraph.cycle(5)
print(C)
S=ptens.subgraph.star(5)
print(S)

M=torch.tensor([[0,1,1],[1,0,1],[1,1,0]],dtype=torch.float)
S=ptens.subgraph.from_matrix(M)
print(S)

ix=torch.tensor([[0,1,2,0,3],[1,2,0,3,0]],dtype=torch.int)
S=ptens.subgraph.from_edge_index(ix)
print(S)

G=ptens.ggraph.random(8,0.5)
S=ptens.subgraph.triangle()
atoms=G.subgraphs(S)
print(atoms)

A=ptens.ptensorlayer1.randn(atoms,3)
print(A)

C=G.cached_subgraph_lists()
print(C)

print("Cache:")
C=ptens_base.subgraph_cache.torch()
for s in C:
    print(s)




