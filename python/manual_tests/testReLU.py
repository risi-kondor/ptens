import torch
import ptens as p

A=p.ptensors0.randn(3,3)
A.requires_grad_()
print(A)

B=p.relu(A,0)
print(B)


