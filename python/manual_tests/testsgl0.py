import torch
import ptens

G=ptens.ggraph.random(5,0.5)
#M=torch.randn(5,3)
#A=ptens.subgraph_layer0.from_matrix(G,M)

A=ptens.subgraph_layer0.sequential(G,3)
M=A.torch()

print(M)
