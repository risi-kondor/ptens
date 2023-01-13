import torch
import ptens as p
N = 50
G = p.graph.random(N,0.1)
x = p.ptensors1.randn([[i] for i in range(N)],64).to('cuda')
x.requires_grad = True
y = torch.rand(N,128).to('cuda')
x = p.transfer1(x,G.nhoods(1),G)
x.backward()
