import torch
import ptens
from torch.autograd.gradcheck import gradcheck

G = ptens.ggraph.random(11, 1.0)

atoms_in = G.subgraphs(ptens.subgraph.edge())
atoms_out = G.subgraphs(ptens.subgraph.trivial())

nc = 2
n=len(atoms_in)
x=torch.randn([2*n,nc])
#x = ptens.ptensorlayer1.sequential(atoms_in, nc) + 1
x.requires_grad_()
#print(x)

def reduction_fn(x):
    zeros = x * 0
    l1loss = torch.nn.L1Loss(reduction="sum")
    return l1loss(x, zeros)

def fn(x):
    #print(len(atoms_in))
    x=ptens.ptensorlayer1.from_matrix(atoms_in,x)
    z = ptens.ptensorlayer1.gather(atoms_out, x)
    s = reduction_fn(z)
    return s

s=fn(x)
s.backward(torch.tensor(1))
g=x.grad
#print(g)

print(gradcheck(fn, (x,), eps= 1e-3, nondet_tol=1e-1))
