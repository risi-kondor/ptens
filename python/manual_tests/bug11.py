import torch
import ptens

# fails on both the cpu and gpu
# fails for ptensors1 and ptensors2, but not ptensors0
num_nodes = 64
channels = 32
values = torch.rand(num_nodes,channels,requires_grad=True)
atoms = [[i] for i in range(num_nodes)]     
m = torch.rand(channels,channels)

x = ptens.ptensors2.from_matrix(values,atoms)
y = (x*m).torch()

y.relu().sum().backward()
