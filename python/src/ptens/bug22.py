import ptens
import ptens_base
import torch

def l1loss_reduction(x):
    zero = torch.zeros_like(x)
    l1_sum = torch.nn.L1Loss(reduction="sum")
    s = l1_sum(x, zero)
    return s

G = ptens.ggraph.from_edge_index(torch.Tensor([[0, 1], [1, 0]]).int())
nc = 2

x_list = []
atoms_out_list = []
for atoms_in in [G.subgraphs(ptens.subgraph.trivial()), G.subgraphs(ptens.subgraph.trivial())]:
    atoms_out_list.append(atoms_in)
    x_list.append((ptens.ptensorlayer0.randn(atoms_in, nc)))

x = ptens.batched_ptensorlayer0.from_ptensorlayers(x_list)
atoms_out = ptens_base.batched_atomspack(atoms_out_list)


x = x.to("cuda")
x.requires_grad_()

def loss_fn(x):
    z = ptens.batched_ptensorlayer0.gather(atoms_out, x)
    s = l1loss_reduction(z)
    return s

s = loss_fn(x)
s.backward(torch.tensor(1))
g = x.grad
print(g)

