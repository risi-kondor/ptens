import torch
import ptens as p

ix=torch.tensor([[0,1,2],[1,2,0]],dtype=torch.int)
d=torch.tensor([4,3,3],dtype=torch.int)

print("--- Triangle ---")
s0=p.subgraph.from_edge_index(ix,3)
print(s0)

print("--- Triangle with degrees ---")
s1=p.subgraph.from_edge_index(ix,degrees=d)
print(s1)
#print(G.subgraphs(s1))

print("Contents of cache:\n")
print(p.subgraph_cache.subgraphs())
