import torch
import ptens as p

pts1 = [p.ptensors0, p.ptensors1, p.ptensors2]
pts2 = [p.ptensors0, p.ptensors1, p.ptensors2]
nc = 2
atoms = [[1],[2,5],[1,2,6]]
fn = p.outer
c = 0
for i in pts1:
    x = i.randn(atoms, nc)
    for j in pts2:
        c+=1
        y = j.randn(atoms, nc)
        z = fn(x,y)
        print("c:i j",c,':',i,j)
        print(z)
