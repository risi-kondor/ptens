import torch
import ptens as p

A=p.ptensors1.randn([[1,2],[2,3],[3]],3)
G=p.graph.random(3,0.5)
G=p.graph.from_matrix(torch.tensor([[1.0,1.0,0],[1.0,0,0],[0,0,1.0]]))
print(A)
print(G)

B=p.unite1(A,G)
print(B)
