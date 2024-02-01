import torch
import ptens as p

import faulthandler

faulthandler.enable()

device = 'cuda'
p.ggraph.random(10,0.5).cache(0)
G = p.batched_ggraph.from_cache([0])

S=p.subgraph.triangle()
print("S is:", S.torch().shape)

# intialize node_rep for the graph

node_rep = p.batched_ptensors0b.from_matrix(torch.arange(30, dtype=torch.float, device=device).reshape(10, -1), [0])

#M = torch.arange(10, dtype=torch.float)
#print("M is:", M)
#test_ptens = p.batched_subgraphlayer1b.gather_from_ptensors(node_rep, G, S)
#print("test_ptens is:", test_ptens.torch().shape)
#test = p.batched_subgraphlayer1b.like(test_ptens, M)
#print("test is:", test.torch().shape)   
#print("test is", test.torch())


A=p.batched_subgraphlayer1b.gather_from_ptensors(node_rep, G,S)
print("A is:" ,A.torch().shape)

L1=p.Autobahn(3,4,S).to(device)

B=L1(A)
print("B is:", B.torch().shape)

pred = p.batched_subgraphlayer1b.gather_from_ptensors(B,G,S)
pred.torch().sigmoid().sum().backward()
print("all good")

