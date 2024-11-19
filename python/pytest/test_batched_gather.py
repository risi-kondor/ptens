import torch
import ptens
import pytest
import ptens_base
from conftest import numerical_grad_sum, numerical_jacobian, get_atomspack_from_graph_factory_list, get_graph_list

from torch.autograd.gradcheck import gradcheck

def get_graph_atomspack_nc(reduction=False):
    def l1loss_reduction(x):
        zero = torch.zeros_like(x)
        l1_sum = torch.nn.L1Loss(reduction="sum")
        s = l1_sum(x, zero)
        return s
        
    ncs = [2]
    reduction_fns = [None]
    if reduction:
        reduction_fns = [
            torch.sum,
            l1loss_reduction,
        ]

    
    graphs = get_graph_list()
    atomspack_factory = get_atomspack_from_graph_factory_list()

    test_list = []

    for g in graphs:
        atompack_list = [factory(g) for factory in atomspack_factory]
        for nc in ncs:
            for reduction_fn in reduction_fns:
                if reduction:
                    test_list.append((g, atompack_list, nc, reduction_fn))
                else:
                    test_list.append((g, atompack_list, nc))

    return test_list


def backprop_sum(unbatch_cls, batched_cls, G, atoms_list, nc, reduction_fn, device, numerical_single_precision_eps):
    x_list = []
    for atoms in atoms_list:
        x = unbatch_cls.sequential(atoms, nc).to(device) + 1
        x_list.append(x)
    x = batched_cls.from_ptensorlayers(x_list)
    x = x.to(device)
    x.requires_grad_()
    # print("x", x)
    atoms2 = ptens_base.batched_atomspack(list(reversed(atoms_list)))
    
    def loss_fn(x):
        z = batched_cls.gather(atoms2, x)
        s = reduction_fn(z)
        return s

    xgrad = torch.autograd.grad(outputs=loss_fn(x), inputs=x)[0]
    # print("xgrad", xgrad)

    fn = lambda x: batched_cls.gather(atoms2, x)
    xgrad2 = numerical_grad_sum(fn, x, numerical_single_precision_eps)
    # print("xgrad2", xgrad2)

    # assert gradcheck(loss_fn, (x,), eps=numerical_single_precision_eps, rtol=1e-2, atol=1e-1, nondet_tol=1e-3)
    
    assert torch.allclose(torch.Tensor(xgrad), torch.Tensor(xgrad2), rtol=1e-1, atol=1e-1)

    
    assert device in str(xgrad.device)



@pytest.mark.parametrize(("G", "atoms_list", "nc", "reduction_fn"), get_graph_atomspack_nc(True))
def test_gather0_sum(G, atoms_list, nc, reduction_fn, device, numerical_single_precision_eps):
    backprop_sum(ptens.ptensorlayer0, ptens.batched_ptensorlayer0, G, atoms_list, nc, reduction_fn, device, numerical_single_precision_eps)

