import torch
import ptens_base
import ptens

G0=ptens.ggraph.random(6,0.5)
G1=ptens.ggraph.random(6,0.5)
G2=ptens.ggraph.random(6,0.5)
G=ptens.batched_ggraph.from_graphs([G0,G1,G2])
print(G)

print(ptens_base.status_str())


