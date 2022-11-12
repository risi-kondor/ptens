import torch
import ptens as p
w = torch.rand(3,2,device='cuda')
b = torch.rand(2,device='cuda')
x = p.ptensors0.randn([[i] for i in range(10)],3).to('cuda')
p.linear(x,w,b)