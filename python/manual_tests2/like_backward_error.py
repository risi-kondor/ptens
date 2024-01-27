import torch
import ptens as p
import torch_geometric.utils as pyg_utils

device = 'cpu'
n = 4
M = torch.tensor([[0,0,0],[1,2,3]]).float()
G = p.ggraph.from_edge_index(M)

edge = p.subgraph.edge()

M = torch.arange(n).view(n,1).float().to(device)
M.requires_grad = True
print(M.shape)
A = p.ptensors0b.from_matrix(M)
print("A\n",A)

C = p.subgraphlayer1b.gather_from_ptensors(A,G,edge)
print("C\n",C)
C_like = p.subgraphlayer1b.like(C,C.torch())
pred = C_like.torch().sum()
print(f"pred: {pred}")
pred.backward()
