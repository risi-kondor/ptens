import torch
import ptens
# hyper params
N = 50
in_channels = 4
out_channels = 4
# creating features
graph = ptens.graph.random(N,0.1)
x = ptens.ptensors1.randn(graph.nhoods(2),in_channels)
x = x.to('cuda')
# creating labels
y = torch.randint(out_channels - 1,(N,),device='cuda')


# creating weights
m = torch.rand(in_channels,out_channels,requires_grad=True,device='cuda')
b = torch.rand(out_channels,requires_grad=True,device='cuda')

# computing forward
x = ptens.linear(x,m,b)
x = ptens.linmaps0(x)
x = x.torch()
x = torch.softmax(x,1)

# computing backward
loss = torch.nn.NLLLoss()
x = loss(x,y)
# V where the code breaks
x.backward()
