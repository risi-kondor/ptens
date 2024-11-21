import torch
import ptens

G = ptens.ggraph.random(3, 1.0)

atoms_in = G.subgraphs(ptens.subgraph.triangle())
atoms_out = G.subgraphs(ptens.subgraph.trivial())

nc = 1
x = ptens.ptensorlayer2.sequential(atoms_in, nc) + 2
#x = x.to('cuda')
x.requires_grad_()
print(x)

z = ptens.ptensorlayer2.gather(atoms_out, x)
print(z)
zero = z.zeros_like()
print(z.device, zero.device)

l1sum = torch.nn.L1Loss(reduction='sum')
def reduction_fn(z):
    return l1sum(zero, z)

# reduction_fn = torch.sum

def fn(x):
    z = ptens.ptensorlayer2.gather(atoms_out, x)
    s = reduction_fn(z)
    return s

s=fn(x)
s.backward(torch.tensor(1))
g=x.grad
print(g)
