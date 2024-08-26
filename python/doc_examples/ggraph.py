import torch
import ptens_base
import ptens

A=torch.tensor([[0,1,0],[1,0,1],[0,1,0]],dtype=torch.int)
G=ptens.ggraph.from_matrix(A)
print(G)

M=(torch.randn(3,2)*10).int()
G=ptens.ggraph.from_matrix(A,labels=M)
print(G)

A=torch.tensor([[0,1,1,2,0,3],[1,0,2,1,3,0]],dtype=torch.int)
G=ptens.ggraph.from_edge_index(A)
print(G)

A=torch.tensor([[0,1,0],[1,0,1],[0,1,0]],dtype=torch.int)
G=ptens.ggraph.from_matrix(A)
print(G)

A=torch.tensor([[0,3,0],[3,0,7],[0,7,0]],dtype=torch.int)
G=ptens.ggraph.from_matrix(A)
print(G)

M=G.adjacency_matrix()
print(M)

G=ptens.ggraph.random(8,0.3)
print(G)




