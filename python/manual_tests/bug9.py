import torch
import ptens as p
x = torch.tensor([[3]],dtype=torch.float,requires_grad=True)
x_1 = p.ptensors0.from_matrix(x)
y_2 = p.ptensors0.mult_channels(x_1,torch.tensor([1],dtype=torch.float))
y_2.backward() 
print(x._grad)
x = torch.tensor([[3]],dtype=torch.float,requires_grad=True)
y_1 = x * 1
y_1.backward()
print(x._grad)
