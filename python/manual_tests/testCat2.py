import torch
import ptens
r'''
Two matching Ptensor layers can be combined with the cat  function:
'''
#A=ptens.ptensors0.randn([[1,2],[2,3],[3]],3)
#B=ptens.ptensors0.randn([[1,2],[2,3],[3]],3)
#C=ptens.concat(A,B)
#print('ptensors0) C=', C)

A=ptens.ptensors1.randn([[1,2],[2,3],[3]],3)
B=ptens.ptensors1.randn([[1,2],[2,3],[3]],3)
A.requires_grad_()
B.requires_grad_()

C=ptens.ptensors1.cat(A,B)
print('ptensors1) C=', C)

out=C.inp(C)
#print(out)
out.backward(torch.tensor(1.0))
print(A.get_grad())
print(B.get_grad())



#A=ptens.ptensors2.randn([[1,2],[2,3],[3]],3)
#B=ptens.ptensors2.randn([[1,2],[2,3],[3]],3)
#C=ptens.concat(A,B)
#print('ptensors2) C=', C)
