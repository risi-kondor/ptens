import torch
import ptens

G=ptens.ggraph.random(3,0.5)
S=ptens.subgraph.triangle()
#M=torch.randn(5,3)
#A=ptens.subgraph_layer0.from_matrix(G,M)

A=ptens.subgraph_layer1.sequential(G,S,[[0,1,2],[0,1,2],[0,1,2]],3)

print(A)
