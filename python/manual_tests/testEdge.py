import torch
import ptens as p
ix=torch.tensor([[1,2,4,5],[2,1,5,4]],dtype=torch.float)
print("ix= ", ix)
G=p.graph.from_edge_index(ix)
print("G= ", G)

ix2=torch.tensor([[0,1],[1,0]],dtype=torch.float)
print("ix2= ", ix2)
H=p.graph.from_edge_index(ix2)
print("H= ", H.obj)
apack=G.subgraphs(H)
print("apack= ", apack)
