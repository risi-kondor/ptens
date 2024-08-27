import torch
import ptens_base as pb
import ptens as p


print("\n----------------------------------")
print(" catomspack")
print("----------------------------------\n")

print("A random compressed atomspack can be defined from a regular atomspack:\n")
a=pb.atomspack.from_list([[1,3,4],[2,5]])
A=pb.catomspack.random(a,4)
print(A)

print("We can also define it from a matrix:\n")
a=pb.atomspack.from_list([[1,3,4],[2,5]])
M=torch.randn(a.nrows1(),4)
A=pb.catomspack(a,M)
print(A)

print("\nWe can retrieve individual bases:\n") 
print(A.basis(1))

print("\nOr all the bases together:\n") 
print(A.torch())

print("\n")







