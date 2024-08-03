import torch
import ptens_base
import ptens

atoms1=ptens_base.atomspack([[1,2,3],[3,5],[2]])
atoms2=ptens_base.atomspack([[3,2],[1,4],[3]])

L=ptens_base.layer_map.overlaps_map(atoms2,atoms1)
print(L)

in_atoms=ptens_base.atomspack.from_list([[1,3,4],[2,5],[0,2]])
out_atoms=ptens_base.atomspack.from_list([[2,4],[3,5],[1]])
L=ptens_base.layer_map.overlaps_map(out_atoms,in_atoms)
A=ptens.ptensorlayer1.randn(in_atoms,3)
print(A)

B=ptens.ptensorlayer1.gather(out_atoms,A,L)
print(B)


A=ptens.ptensorlayer1.randn([[1,3,4],[2,5],[0,2]],3)
B=ptens.ptensorlayer1.gather([[2,4],[3,5],[1]],A)

