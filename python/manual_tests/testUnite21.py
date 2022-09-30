import torch
import ptens as p

M=torch.tensor([[1.0],[2.0]])
A=p.ptensors0.from_matrix(M)
A.requires_grad_()
print(A)

G=p.graph.from_matrix(torch.tensor([[1.0,1.0],[1.0,1.0]]))
print(G)

B=p.unite2(A,G)
print(B)

loss=B.inp(B)
print(loss)

loss.backward(torch.tensor(1.0))

print(A.get_grad())

