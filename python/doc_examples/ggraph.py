import torch
import ptens_base
import ptens

A=torch.tensor([[0,1,0],[1,0,1],[0,1,0]],dtype=torch.float32)
G=ptens.ggraph.from_matrix(A)
print(G)

A=torch.tensor([[0,1,1,2,0,3],[1,0,2,1,3,0]],dtype=torch.float32)
G=ptens.ggraph.from_edge_index(A)
print(G)

M=G.adjacency_matrix()
print(M)

G=ptens.ggraph.random(8,0.3)
print(G)




