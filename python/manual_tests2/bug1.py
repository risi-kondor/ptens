import numpy as np
import torch
import ptens

print("A")
A=ptens.ptensor0.randn([2],5, device="gpu")
A = A.to("cpu")
print(A)

B = ptens.ptensors0b.randn([[2]],5, device="gpu")
B = B.to("cpu")
print("B is down as well")
print(B)

