import torch
import ptens
from ptens_base import atomspack

G = ptens.graph.from_edge_index(torch.tensor([[0,1],[1,0]],dtype=torch.float))
print(G)
source = atomspack([[0,1]])
sink = atomspack([[0],[1]])
#
target_domains = G.subgraphs(G)
substruct_map = ptens.graph.overlaps(sink,source)
#print(substruct_map)
#

x = ptens.ptensors1.randn(source,16)
x.requires_grad = True
#x = x.to('cuda')

s = ptens.ptensors1.transfer1(x,sink,substruct_map)

s = s.torch()
#s.backward(s)
loss = torch.nn.functional.l1_loss(s,torch.zeros_like(s))
loss.backward()

