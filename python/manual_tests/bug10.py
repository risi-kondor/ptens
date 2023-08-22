import torch
import ptens

num_nodes = 10
num_channels = 64
values = torch.empty(num_nodes,num_channels,requires_grad=True)
atoms = [[i] for i in range(num_nodes)]

x = ptens.ptensors0.from_matrix(values.cuda(),atoms)

# Only breaks when done on the GPU and concatenating on dimension -1.
# It also doesn't care if we are concatenating a tensor to itself or two distinct tensors.
# However, calling '.torch()' twice as done below is necessary.
y = torch.cat([x.torch(),x.torch()],-1)
y.norm().backward()
#y.relu().sum().backward()

