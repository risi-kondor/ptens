import torch
import ptens as p

atoms0=[[1,2],[4],[0,2,5],[3,1]]
atoms1=[[1],[2,4],[1,2,5],[0,1]]
atoms=[atoms0,atoms1,atoms1]

M=torch.randn([24,3])

x=p.batched_ptensors1b.randn(atoms,3)
print(x)
print(x.torch())

A=p.batched_ptensors1b.from_matrix(M,atoms)
print(A)
print(A.torch())

#B=A.relu(A)
#print(B)

M2=torch.randn([3,3])
print(M2)
print(A*M2)


