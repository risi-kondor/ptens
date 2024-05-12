import torch
import ptens

ix=torch.tensor([[1,2,4,5],[2,1,5,4]],dtype=torch.float)
G=ptens.graph.from_edge_index(ix)
print("G=", G)

ix2=torch.tensor([[0,1],[1,0]],dtype=torch.float)
H=ptens.graph.from_edge_index(ix2)
apack=G.subgraphs(H)

print("apack=", apack)
