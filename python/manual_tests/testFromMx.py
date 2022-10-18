import torch
import ptens as p

A=p.ptensors2.randn([[1,2,3],[3,5],[2]],3)
print(A)

At=A.torch()
print(At)

B=B=p.ptensors2.from_matrix(At,[[1,2],[1,2,3],[1]])
print(B)
