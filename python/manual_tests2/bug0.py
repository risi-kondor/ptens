import torch
import ptens as p
G=p.ggraph.random(5,0.8)
S=p.subgraph.triangle()
A=p.subgraphlayer1b.sequential(G,S,3)
print(A)
B=p.subgraphlayer1b.gather(A,S)
print(B)
