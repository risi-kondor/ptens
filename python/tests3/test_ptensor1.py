import torch
import ptens_base as pb
import ptens as p

print("We can define a random first order P-tensor:\n")
A=p.ptensor1.randn([2,3,5],3)
print(A.__repr__(),"\n")
print(A)

print("Or define it from a torch tensor:\n")
M=torch.randn([3,3])
A=p.ptensor1.from_tensor([2,3,5],M)
print(A)

V=torch.relu(A)
print(V)

print("If two ptensors have the same reference domain,\n it is possible to do arithmetic on them:\n")
# Unfortunately these have to be added manually one-by-one
B=p.ptensor1.randn([2,3,5],3)
print(A+B)


print("\n---------")
print(" Linmaps")
print("---------\n")


print("The linmaps from a 0th order P-tensor broadcasts along the atom dimension:\n")
A=p.ptensor0.randn([2,3,5],3)
print(p.ptensor1.linmaps(A))

print("The linmaps from a 1st order P-tensor consists of two different maps:\n")
A=p.ptensor1.randn([2,3,5],3)
print(p.ptensor1.linmaps(A))

print("The linmaps from a 2nd order P-tensor consists of 5 different maps:\n")
A=p.ptensor2.randn([2,3,5],3)
print(p.ptensor1.linmaps(A))

