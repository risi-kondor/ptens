import torch
import ptens_base as pb
import ptens as p

print("\n----------------------------------")
print(" ptensorlayer1")
print("----------------------------------\n")

print("A ptensorlayer1 is a collection of first order\nptensors stored in a single matrix.\n")

print("We can define a random zeroth order P-tensor:\n")
atoms=pb.atomspack.from_list([[1,3,4],[2,5],[0,2]])
A=p.ptensorlayer1.randn(atoms,3)
print(A.__repr__(),"\n")
print(A)

print("Or define it from a torch tensor:\n")
M=torch.randn([atoms.nrows1(),3])
A=p.ptensorlayer1.from_matrix(atoms,M)
print(A)

print("If two ptensors have the same reference domain,\nwe can do arithmetic on them:\n")
# Unfortunately these have to be added manually one-by-one
B=p.ptensorlayer1.randn(atoms,3)
print(A+B)

