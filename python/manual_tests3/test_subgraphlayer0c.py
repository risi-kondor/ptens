import torch
import ptens as p
import ptens_base as pb

G=p.ggraph.random(8,0.5)
print(G)

#S=p.subgraph.edge()
S=p.subgraph.triangle()

X=p.subgraphlayer0c.zeros(G,S,3)
print(X)

print("\n\n")

