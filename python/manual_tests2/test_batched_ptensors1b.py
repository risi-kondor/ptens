import torch
import ptens as p
import ptens_base

atoms0=[[1,2],[4],[0,2,5],[3,1]]
atoms1=[[1],[2,4],[1,2,5],[0,1]]
atoms=[atoms0,atoms1,atoms1]

M=torch.randn([24,3])

x=p.batched_ptensors1b.randn(atoms,3)
print(x)
print(x.torch())

A=p.batched_ptensors1b.from_matrix(M,atoms)
print(A[0])
print(A.torch())

s=ptens_base.batched_atomspack([[[1,2],[3,4]],[[2,3]],[[1,2]]])

B=p.batched_ptensors2b.gather(A,s);
print(B)
exit()
#B=A.relu(A)
#print(B)

M2=torch.randn([3,3])
print(M2)
print(A*M2)


