import torch
import ptens as p
N = 2
num_features = 2
num_classes = 4
x = p.ptensors1.randn([[i] for i in range(N)],num_features)
y = torch.randint(num_classes,size=(N,))

w = torch.nn.parameter.Parameter(torch.rand(num_features,num_classes))
b = torch.nn.parameter.Parameter(torch.rand(num_classes))

loss = torch.nn.NLLLoss()
optim = torch.optim.Adam([w,b],0.9)

optim.zero_grad()
a = p.linear(x,w,b)
a = p.linmaps0(a)
a = a.torch()
loss(a,y).backward()
optim.step()
x = p.linear(x,w,b)