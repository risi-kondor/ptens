import torch
import ptens_base as pb
import ptens as p

print("\n-------------------------------------")
print (" batched_ptensorlayer2")
print("-------------------------------------\n")

print("A batched_ptensorlayer2 stores a batch of ptensorlayer2 objects.\n\n")

print("We can construct it from a collection of ptensor layers:\n")
subatoms=pb.atomspack.from_list([[1,3,4],[2,5],[0,2]])
a=p.ptensorlayer2.randn(subatoms,3)
A=p.batched_ptensorlayer2.from_ptensorlayers([a,a,a])
print(A)

print("Or from a torch tensor:\n")
atoms=pb.batched_atomspack([subatoms,subatoms,subatoms])
M=torch.randn([atoms.nrows2(),3])
A=p.batched_ptensorlayer2.from_matrix(atoms,M)
print(A)

print("If two batched P-tensor layers have the same reference domains,\nwe can do arithmetic on them:\n")
# Unfortunately these have to be added manually one-by-one
B=p.batched_ptensorlayer2.randn(atoms,3)
print(A+B)


print("\n---------")
print(" Linmaps")
print("---------\n")

print("The linmaps from a 0th batched layer consist of two different maps:\n")
A0=p.batched_ptensorlayer0.randn(atoms,3)
B=p.batched_ptensorlayer2.linmaps(A0)
print(B)

print("The linmaps from a 1st batched layer consist of five different maps:\n")
A1=p.batched_ptensorlayer1.randn(atoms,3)
B=p.batched_ptensorlayer2.linmaps(A1)
print(B)

print("The linmaps from a 1st batched layer consist of 15 different maps:\n")
A2=p.batched_ptensorlayer2.randn(atoms,3)
print(p.batched_ptensorlayer2.linmaps(A2))


print("\n---------")
print(" Gather")
print("---------\n")


subatoms2=pb.atomspack.random(5,5,0.6)
atoms2=pb.batched_atomspack([subatoms2,subatoms2,subatoms2])

B0=p.batched_ptensorlayer2.gather(atoms2,A0)
print(B0)

B1=p.batched_ptensorlayer2.gather(atoms2,A1)
print(B1)

B2=p.batched_ptensorlayer2.gather(atoms2,A2)
print(B2)

