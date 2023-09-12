import torch
from ptens import *

G=ggraph.random(5,0.5)
print(G)

M=torch.randn(5,3)
f0=subgraphlayer0.from_matrix(G,M)
print(f0)

edge=subgraph.edge()
print(edge)

f1=subgraphlayer1.gather(f0,edge)
f1.requires_grad_()
print(f1)

f2=f1.relu(0)
print(f2)

out=f2.inp(f2)
print(out)

out.backward(torch.tensor(1.0))
print(f1.get_grad())




