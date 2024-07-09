import torch
import ptens_base as pb
import ptens as p

print("\n-------------------------------------")
print (" ptensor0")
print("-------------------------------------\n")


print("We can define a random zeroth order P-tensor:\n")
A=p.ptensor0.randn([2,3],3)
print(A.__repr__(),"\n")
print(A)

print("Or define it from a torch tensor:\n")
M=torch.randn([3])
A=p.ptensor0.from_tensor([2,3],M)
print(A)

print("If two ptensors have the same reference domain,\nit is possible to do arithmetic on them:\n")
# Unfortunately these have to be added manually one-by-one
B=p.ptensor0.randn([2,3],3)
print(A+B)


print("\n---------")
print(" Linmaps")
print("---------\n")


print("The linmaps from a 0th order P-tensor is just the identity:\n")
A=p.ptensor0.randn([2,3,5],3)
print(p.ptensor0.linmaps(A))

print("The linmaps from a 1st order P-tensor sums along the atoms dimension:\n")
A=p.ptensor1.randn([2,3,5],3)
print(p.ptensor0.linmaps(A))

print("The linmaps from a 2nd order P-tensor consists of")
print("(a) summing along both atoms dimensions")
print("(b) summing along the diagonal:\n")
A=p.ptensor2.randn([2,3,5],3)
print(p.ptensor0.linmaps(A))
