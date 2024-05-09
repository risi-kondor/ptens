import torch
import ptens as p

atoms=[[1,2],[4],[0,2,5],[3,1]]
M=torch.randn([8,3])

x=p.ptensors1b.randn(atoms,3)
print(x.torch())

A=p.ptensors1b.from_matrix(M,atoms)
print(A)
print(A.torch())

print(A.get_atoms().torch()[1])

#B=A.relu(A)
#print(B)

M2=torch.randn([3,3])
print(M2)
print(A*M2)

S=torch.randn([3])

