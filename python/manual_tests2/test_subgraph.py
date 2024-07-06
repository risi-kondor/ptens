import torch
import ptens_base as pb
import ptens as p


ix=torch.tensor([[0,1,2],[1,2,0]],dtype=torch.float)
d=torch.tensor([4,3,3],dtype=torch.float)

G=p.ggraph.random(8,0.5)
print(G)

print("--- Triangle ---")
s0=p.subgraph.from_edge_index(ix)
print(s0)
print(G.subgraphs(s0))

print("--- Triangle with degrees ---")
s1=p.subgraph.from_edge_index(ix,degrees=d)
print(s1)
print(G.subgraphs(s1))

print("--- Cache ---")
print(p.subgraph.cached())
