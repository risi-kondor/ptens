import torch
import ptens_base
import ptens


A=ptens.ptensor0.sequential([1,2,3],5)
print(A)

B=ptens.ptensor0.linmaps(A)
print(B)

A=ptens.ptensor1.sequential([1,2,3],3)
print(A)

B=ptens.ptensor0.linmaps(A)
print(B)

A=ptens.ptensor2.sequential([1,2,3],3)
print(A)

B=ptens.ptensor0.linmaps(A)
print(B)

A=ptens.ptensor0.sequential([1,2,3],3)
print(A)

B=ptens.ptensor1.linmaps(A)
print(B)

A=ptens.ptensor1.sequential([1,2,3],3)
print(A)

B=ptens.ptensor1.linmaps(A)
print(B)

A=ptens.ptensor2.sequential([1,2,3],3)
print(A)

B=ptens.ptensor1.linmaps(A)
print(B)

A=ptens.ptensorlayer1.randn([[1,2],[2,3],[4]],2)
print(A)

B=ptens.ptensorlayer1.linmaps(A)
print(B)


