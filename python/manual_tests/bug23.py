import torch
import ptens
import ptens_base
from torch.autograd.gradcheck import gradcheck



graph_list = []
for i in range(5): # 5 is random, in testing most of the time I get an error after 1
    graph_list.append(ptens.ggraph.from_edge_index(torch.Tensor([[0, 1], [1, 0]]).int()))

def trivial(g):
    return g.subgraphs(ptens.subgraph.trivial())
factory_list = [trivial]

def l1loss_reduction(x):
    zero = torch.zeros_like(x)
    l1_sum = torch.nn.L1Loss(reduction="sum")
    s = l1_sum(x, zero)
    return s


def get_batched_layer(unbatched_cls, cls, G, nc):
    x_list = []
    atoms_out_list = []
    for atoms_factory in factory_list:
        atoms_in = atoms_factory(G)
        atoms_out_list.append(G.subgraphs(ptens.subgraph.trivial()))
        x_list.append(unbatched_cls.sequential(atoms_in, nc)+1)

    x = cls.from_ptensorlayers(x_list)
    atoms2 = ptens_base.batched_atomspack(atoms_out_list)
    return x, atoms2

def test(G, device, last):
    cls = ptens.batched_ptensorlayer0
    unbatched_cls = ptens.ptensorlayer0
    x, atoms_out = get_batched_layer(unbatched_cls, cls, G, 2)

    x = x.to(device)
    x.requires_grad_()
    
    def loss_fn(x):
        z = cls.gather(atoms_out, x)
        s = l1loss_reduction(z)
        return s


    s = loss_fn(x)
    s.backward(torch.tensor(1))
    xgrad = x.grad
    
    print("xgrad", torch.Tensor(xgrad), "\n")
    if last is not None:
        assert torch.allclose(last, xgrad)

    return torch.Tensor(xgrad)


last = None
for g in graph_list:
    last = test(g, 'cuda', last)
    
