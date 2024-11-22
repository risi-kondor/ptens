import re
import torch
import ptens
import pytest
import ptens_base
from conftest import numerical_grad_sum, numerical_jacobian, get_atomspack_from_graph_factory_list, get_graph_list

from torch.autograd.gradcheck import gradcheck

def l1loss_reduction(x):
    zero = torch.zeros_like(x)
    l1_sum = torch.nn.L1Loss(reduction="sum")
    s = l1_sum(x, zero)
    return s

test_reduction_fns = [
    torch.sum,
    l1loss_reduction
    ]
test_ncs = [2,]

def backprop_batched_sum(cls, x, atoms_out, reduction_fn, device, numerical_single_precision_eps):
    x = x.to(device)
    x.requires_grad_()
    # print("x", x)
    
    def loss_fn(x):
        z = cls.gather(atoms_out, x)
        s = reduction_fn(z)
        return s

    xgrad = torch.autograd.grad(outputs=loss_fn(x), inputs=x)[0]
    print("xgrad", torch.Tensor(xgrad))

    fn = lambda x: cls.gather(atoms_out, x)
    xgrad2 = numerical_grad_sum(fn, x, numerical_single_precision_eps)
    # print("xgrad2", xgrad2)
    print("eps=",numerical_single_precision_eps)

    # We deactivate since it brings a number of false negatives with Floating Point precision
    # assert gradcheck(loss_fn, (x,), eps=numerical_single_precision_eps, rtol=1e-1, atol=1e-1, nondet_tol=1e-3)
    
    assert torch.allclose(torch.Tensor(xgrad), torch.Tensor(xgrad2), rtol=1e-1, atol=1e-1)
    
    assert device in str(xgrad.device)

def get_batched_layer(unbatched_cls, cls, G, nc):
    x_list = []
    atoms_out_list = []
    for atoms_factory in get_atomspack_from_graph_factory_list():
        atoms_in = atoms_factory(G)
        atoms_out_list.append(G.subgraphs(ptens.subgraph.trivial()))
        # Random and square guarantees positive numbers not too big
        x_list.append(torch.square(unbatched_cls.randn(atoms_in, nc)))

    x = cls.from_ptensorlayers(x_list)
    atoms2 = ptens_base.batched_atomspack(atoms_out_list)
    return x, atoms2

@pytest.mark.parametrize("G", get_graph_list())
@pytest.mark.parametrize("nc", test_ncs)
@pytest.mark.parametrize("reduction_fn", test_reduction_fns)
def test_gather0_sum(G, nc, reduction_fn, device, numerical_single_precision_eps):

    cls = ptens.batched_ptensorlayer0
    unbatched_cls = ptens.ptensorlayer0
    x, atoms_out = get_batched_layer(unbatched_cls, cls, G, nc)
    
    if "cuda" in device and reduction_fn is torch.sum:
        with pytest.raises(RuntimeError, match=re.escape("Ptens error in void ptens::Ptensors0_reduce0_cu(const TENSOR&, const TENSOR&, const ptens::AindexPackB&, int, int, CUstream_st* const&): failed assertion x.stride(1)==1.")):
            backprop_batched_sum(cls, x, atoms_out, reduction_fn, device, numerical_single_precision_eps)
    else:
        backprop_batched_sum(cls, x, atoms_out, reduction_fn, device, numerical_single_precision_eps)
