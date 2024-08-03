import torch
import ptens_base
import ptens

subatoms=ptens_base.atomspack.from_list([[1,3,4],[2,5],[0,2]])
a=ptens.ptensorlayer1.randn(subatoms,3)
A=ptens.batched_ptensorlayer1.from_ptensorlayers([a,a,a])
print(A.__repr__())
print(A)

atoms=ptens_base.batched_atomspack([subatoms,subatoms,subatoms])
M=torch.randn([atoms.nrows1(),3])
A=ptens.batched_ptensorlayer1.from_matrix(atoms,M)
print(A)

a1=ptens_base.atomspack.random(3,6,0.5)
a2=ptens_base.atomspack.random(4,6,0.5)
batched_atoms1=ptens_base.batched_atomspack([a1,a1,a1]) #TODO: random constructor
batched_atoms2=ptens_base.batched_atomspack([a2,a2,a2])
L=ptens_base.batched_layer_map.overlaps_map(batched_atoms2,batched_atoms1)
print(L) # TODO!!

print(333)
A=ptens.batched_ptensorlayer1.randn(batched_atoms1,3)
B=ptens.batched_ptensorlayer1.gather(batched_atoms2,A) 
print(B)



