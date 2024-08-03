import torch
import ptens_base
import ptens

atoms=ptens_base.atomspack([[1,2,3],[3,5],[2]])
print(atoms)

A=ptens.ptensorlayer1.randn(atoms,3)
print(A)

A=ptens.ptensorlayer1.randn([[1,2,3],[3,5],[2]],3)
print(A)

print(torch.Tensor(A))

M=torch.randn(6,3)
A=ptens.ptensorlayer1.from_matrix([[1,2,3],[3,5],[2]],M)
print(A)

print(A[1])

A=ptens.ptensorlayer1.randn([[1,2,3],[3,5],[2]],3)
B=torch.relu(A)
print(B)

A=ptens.ptensorlayer1.randn([[1,2,3],[3,5],[2]],3)
print(torch.norm(A))

print(A.atoms[1])

print(A.atoms.nrows1(1))
print(A.atoms.row_offset1(1))


