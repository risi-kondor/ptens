import torch
import ptens_base
import ptens

A=ptens.ptensor0.randn([2],5)
print(A)

B=ptens.ptensor1.randn([1,2,3],5)
print(B)

C=ptens.ptensor2.randn([1,2,3],5)
print(C)

A=ptens.ptensor1.sequential([1,2,3],5)
print(A)

#A=ptens.ptensor1.sequential([1,2,3],5,device='gpu')

#B=A.to('cpu')
