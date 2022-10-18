import torch
import ptens as p

N=4

M=torch.tensor([[1.0],[2.0]])
A=p.ptensors0.randn(N,1)
A.requires_grad_()
print(A)

#G=p.graph.from_matrix(torch.tensor([[1.0,1.0],[1.0,1.0]]))
G=p.graph.random(N,0.3)
print(G)

B=p.unite1(A,G)
print(B)

#loss=B.inp(B)
#print(loss)

#loss.backward(torch.tensor(1.0))

#print(A.get_grad())
