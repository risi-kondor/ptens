import torch
import ptens as p

atoms=[[1,2],[4],[0,2,5],[3,1]]
M=torch.randn([4,3])

print(p.tlayer.randn(1,atoms,3))

A=p.tlayer.from_matrix(0,M,atoms)
print(A)
print(A.torch())

#B=A.relu(A)
#print(B)

M2=torch.randn([3,3])
print(M2)
print(A*M2)

S=torch.randn([3])

