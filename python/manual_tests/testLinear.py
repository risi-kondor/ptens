import torch
import ptens as p

A=p.ptensors2.randn([[1,2,3],[3,5],[2]],3)
print(A)

w=torch.tensor([[1.0,0.0],[0.0,1.0],[0.0,0.0]])
b=torch.tensor([1.0,2.0,3.0])
B=p.linear(A,w,b)
print(B)

