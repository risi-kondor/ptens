import torch
import ptens

G = ptens.ggraph.random(2, 1.0)

atoms_in = G.subgraphs(ptens.subgraph.trivial())
atoms_out = G.subgraphs(ptens.subgraph.trivial())

nc = 2
x = ptens.ptensorlayer2.sequential(atoms_in, nc) + 1

#x = x.to('cuda')
x.requires_grad_()


xjac = torch.autograd.functional.jacobian(lambda x: x.gather(atoms_out, x), x)
print(xjac.size())
