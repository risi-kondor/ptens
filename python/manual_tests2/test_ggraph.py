import torch
import ptens as p
import ptens_base

G=p.ggraph.random(8,0.5)
print(G)

#S=p.subgraph.edge()
S=p.subgraph.triangle()

A=G.subgraphs(S)
print(A)

