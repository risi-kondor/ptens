import torch
import ptens as p

G=p.ggraph.random(10,0.5)
S=p.subgraph.triangle()

A=p.subgraphlayer1b.randn(G,S,3)
print(A)

L1=p.Autobahn(3,4,S)

B=L1(A)
print(B)
