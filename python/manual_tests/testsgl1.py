import torch
import ptens


G=ptens.ggraph.random(5,0.5)
M=torch.randn(5,3)
f0=ptens.subgraph_layer0.from_matrix(G,M)
print(f0)

S=ptens.subgraph.trivial()
f1=ptens.subgraph_layer0.gather(f0,S)
f1.obj.gather_back(f0.obj)

print(f0)
