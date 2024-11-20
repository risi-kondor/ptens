import torch
import ptens

G = ptens.ggraph.random(2, 1.0)

atoms_in = G.subgraphs(ptens.subgraph.trivial())
atoms_out = G.subgraphs(ptens.subgraph.trivial())

nc = 2
x = ptens.ptensorlayer2.sequential(atoms_in, nc) + 2
x.requires_grad_()

#z = ptens.ptensorlayer2.gather(atoms_out, x)

l1sum = torch.nn.L1Loss(reduction='sum')
def reduction_fn(z):
    zero = z.zeros_like()
    return l1sum(zero, z)

# reduction_fn = torch.sum

def fn(x):
    z = ptens.ptensorlayer2.gather(atoms_out, x)
    s = reduction_fn(z)
    return s

s=fn(x)
print(s)

s.backward(torch.tensor(1))
g=x.grad
print(g)
#exit()

