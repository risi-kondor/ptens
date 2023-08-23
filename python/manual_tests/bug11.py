import torch
import ptens

# fails on both the cpu and gpu
# fails for ptensors1 and ptensors2, but not ptensors0
num_nodes = 2
channels = 10
values = torch.rand(num_nodes,channels,requires_grad=True)
atoms = [[i] for i in range(num_nodes)]     
m = torch.rand(channels,channels)

x = ptens.ptensors2.from_matrix(values.cuda(),atoms)
#y = (x*m).torch()

z=torch.tensor([3.0]).cuda()
y=x.scale(z)

print(x)
print(y)

y.relu().sum().backward()
