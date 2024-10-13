import ptens
import torch
A=ptens.ptensorlayer1.randn([[1,2,3],[3,5],[2]],3)
B=torch.relu(A)
A + B
C = ptens.batched_ptensorlayer1.from_ptensorlayers([A,A,A])
C
D = ptens.batched_ptensorlayer1.from_ptensorlayers([B,B,B])
print((C+D).__repr__())
