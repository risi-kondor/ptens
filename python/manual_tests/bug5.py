import torch
import ptens

in_channels = 512
out_channels = 256
N = 60

G = ptens.graph.random(N,0.05)
x = ptens.ptensors1.randn(G.nhoods(1),in_channels).to('cuda')
w = torch.rand(in_channels,out_channels,dtype=torch.float32,device='cuda')
b = torch.rand(out_channels,dtype=torch.float32,device='cuda')
while True:
  ptens.linear(x,w,b)
