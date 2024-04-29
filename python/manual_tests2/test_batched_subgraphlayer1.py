import torch
import ptens as p

p.ggraph.random(5,0.5).cache(0)
p.ggraph.random(5,0.5).cache(1)
p.ggraph.random(5,0.5).cache(2)

#print(p.ggraph.from_cache(0))
#print(p.ggraph.from_cache(1))

S=p.subgraph.edge()
G=p.batched_ggraph.from_cache([0,1,2,2])
print(G)

A=torch.randn([20,3])
A.requires_grad_()

X=p.batched_ptensors0b.from_matrix(A,[5,5,5,5])
#print(X)

Y=p.batched_subgraphlayer1b.gather_from_ptensors(X,G,S)
print(Y)

Z=p.batched_subgraphlayer1b.linmaps(Y)
print(Z)
print("------")

U=Z.torch()

#U.backward(U)
#print(A.grad)
