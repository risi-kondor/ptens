import torch
import ptens as p
import ptens_base as pb


atomsp=pb.atomspack.from_list([[1,2],[4],[0,2,5],[3,1]])
print(atomsp)

A=p.ptensorlayer0c.randn(atomsp,3)
print(A)

#B=A+A
#print(B)

#W=torch.randn([3,5])
#print(p.mm(B,W))

A=p.ptensorlayer1c.randn(atomsp,3)
print(A)
