import torch
import ptens as p

N = 64
adj_matrix = torch.zeros(N,N,dtype=torch.float32)
adj_matrix[:,0] = 1
adj_matrix[0,:] = 1
adj_matrix[0,0] = 0

G = p.graph.from_matrix(adj_matrix)
x = p.ptensors1.randn(G.nhoods(1),4)
print(x)

x = x.to('cuda')
x = p.unite1(x,G)

print(x)
