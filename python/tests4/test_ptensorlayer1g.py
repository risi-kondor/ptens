import torch
import ptens_base as pb
import ptens as p

device='cpu'

atoms=pb.atomspack.from_list([[1,3,4],[2,5],[0,2]])
atoms2=pb.atomspack.random(5,5,0.6)

A0=p.ptensorlayer0.randn(atoms,3)
A1=p.ptensorlayer1.randn(atoms,3)
A2=p.ptensorlayer2.randn(atoms,3)

B0=p.ptensorlayer1.gather(atoms2,A0)
B1=p.ptensorlayer1.gather(atoms2,A1)
B2=p.ptensorlayer1.gather(atoms2,A2)

A0g=A0.to(device)
A1g=A1.to(device)
A2g=A2.to(device)
print(A0)
print(A0-A0)
C=A0-A0
print(A0.atoms)
print(C.atoms)
print(C)

B0g=p.ptensorlayer1.gather(atoms2,A0g)
B1g=p.ptensorlayer1.gather(atoms2,A1g)
B2g=p.ptensorlayer1.gather(atoms2,A2g)

#print(torch.norm(torch.tensor(B0g-B0)))
print(torch.norm(B0g-B0))
print(torch.norm(B1g-B1))
print(torch.norm(B2g-B2))

