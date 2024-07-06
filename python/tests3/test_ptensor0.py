import torch
import ptens_base as pb
import ptens as p

print("We can define a random zeroth order P-tensor:\n")
A=p.ptensor0.randn([2,3],3)
print(A.__repr__(),"\n")
print(A)

print("Or define it from a torch tensor:\n")
M=torch.randn([3])
A=p.ptensor0.from_matrix([2,3],M)
print(A)

print("If two ptensors have the same reference domain,\nit is possible to do arithmetic on them:\n")
# Unfortunately these have to be added manually one-by-one
B=p.ptensor0.randn([2,3],3)
print(A+B)

