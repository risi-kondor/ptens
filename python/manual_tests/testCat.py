import torch
import ptens
r'''
Two matching Ptensor layers can be concatenated along their channel dim:
'''
A=ptens.ptensors0.randn([[1,2],[2,3],[3]],3)
B=ptens.ptensors0.randn([[1,2],[2,3],[3]],3)
C=ptens.concat(A,B)
print('ptensors0) C=', C)

A=ptens.ptensors1.randn([[1,2],[2,3],[3]],3)
B=ptens.ptensors1.randn([[1,2],[2,3],[3]],3)
C=ptens.concat(A,B)
print('ptensors1) C=', C)

A=ptens.ptensors2.randn([[1,2],[2,3],[3]],3)
B=ptens.ptensors2.randn([[1,2],[2,3],[3]],3)
C=ptens.concat(A,B)
print('ptensors2) C=', C)
