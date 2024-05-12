import torch
import ptens as p


ix=torch.tensor([[0,1,2],[1,2,0]],dtype=torch.float)
d=torch.tensor([0,1,2],dtype=torch.float)

s0=p.subgraph.from_edge_index(ix)
print(s0)

s1=p.subgraph.from_edge_index(ix,degrees=d)
print(s1)

print("Cache:")
print(p.subgraph.cached())
