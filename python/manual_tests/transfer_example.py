import torch
import ptens as p

A=p.ptensors1.randn([[1,2],[3]],3)
G=p.graph.from_matrix(torch.ones(3,2))
print(A)

B=p.transfer1(A,[[1],[2,3],[1,3]],G)
print(B)
