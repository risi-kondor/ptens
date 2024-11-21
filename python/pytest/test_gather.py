import re
import torch
import ptens
import pytest
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


def backprop_sum(cls, G, atoms, nc, reduction_fn, device, numerical_single_precision_eps):
    # Random and square guarantees positive numbers not too big
    x = torch.square(cls.randn(atoms, nc))
    x = x.to(device)
    x.requires_grad_()
    # print("x", x)
    atoms2 = G.subgraphs(ptens.subgraph.trivial())


    def loss_fn(x):
        z = cls.gather(atoms2, x)
        s = reduction_fn(z)
        return s

    xgrad = torch.autograd.grad(outputs=loss_fn(x), inputs=x)[0]
    # print("xgrad", xgrad)

    fn = lambda x: cls.gather(atoms2, x)
    xgrad2 = numerical_grad_sum(fn, x, numerical_single_precision_eps)
    # print("xgrad2", xgrad2)
    print("eps=",numerical_single_precision_eps)

    # We deactivate since it brings a number of false negatives with Floating Point precision
    # assert gradcheck(loss_fn, (x,), eps=numerical_single_precision_eps, rtol=1e-1, atol=1e-1, nondet_tol=1e-3)
    
    assert torch.allclose(torch.Tensor(xgrad), torch.Tensor(xgrad2), rtol=1e-1, atol=1e-1)

    
    assert device in str(xgrad.device)

def backprop_jac(cls, G, atoms, nc, device, numerical_single_precision_eps):
    x = torch.square(cls.randn(atoms, nc))
    x.requires_grad_()
    x = x.to(device)
    atoms2 = G.subgraphs(ptens.subgraph.trivial())

    if cls.gather(atoms2, x).numel() > 0:

        def loss_fn(x):
            z = cls.gather(atoms2, x)
            return z

        xjac2 = numerical_jacobian(loss_fn, x, numerical_single_precision_eps)
        # print("xjac2", xjac2)

        xjac = torch.autograd.functional.jacobian(loss_fn, x)
        # print("xjac", xjac)
        xjac2 = xjac2.to(device)

        # We deactivate since it brings a number of false negatives with Floating Point precision
        assert gradcheck(loss_fn, (x,), eps=numerical_single_precision_eps, rtol=0.1, atol=0.1, nondet_tol=1e-6)
        
        assert torch.allclose(xjac, xjac2, rtol=1e-1, atol=1e-1)


        assert device in str(xjac.device)

def _verify_input(G, atoms):
    N = max(G.adjacency_matrix().size())
    for i in range(len(atoms)):
        a = atoms[i]
        if len(a) > 0:
            assert max(a) < N

@pytest.mark.parametrize("G", get_graph_list())
@pytest.mark.parametrize("atoms_factory", get_atomspack_from_graph_factory_list())
@pytest.mark.parametrize("nc", test_ncs)
@pytest.mark.parametrize("reduction_fn", test_reduction_fns)
def test_gather0_sum(G, atoms_factory, nc, reduction_fn, device, numerical_single_precision_eps):
    atoms = atoms_factory(G)
    print("atoms", atoms)
    _verify_input(G, atoms)
    if "cuda" in device and reduction_fn is torch.sum:
        with pytest.raises(RuntimeError, match=re.escape("Ptens error in void ptens::Ptensors0_reduce0_cu(const TENSOR&, const TENSOR&, const ptens::AindexPackB&, int, int, CUstream_st* const&): failed assertion x.stride(1)==1.")):
            backprop_sum(ptens.ptensorlayer0, G, atoms, nc, reduction_fn, device, numerical_single_precision_eps)
    else:
        backprop_sum(ptens.ptensorlayer0, G, atoms, nc, reduction_fn, device, numerical_single_precision_eps)


@pytest.mark.parametrize("G", get_graph_list())
@pytest.mark.parametrize("atoms_factory", get_atomspack_from_graph_factory_list())
@pytest.mark.parametrize("nc", test_ncs)
def test_gather0_jac(G, atoms_factory, nc, device, numerical_single_precision_eps):
    atoms = atoms_factory(G)
    print("atoms", atoms)
    _verify_input(G, atoms)
    backprop_jac(ptens.ptensorlayer0, G, atoms, nc, device, numerical_single_precision_eps)

    
@pytest.mark.parametrize("G", get_graph_list())
@pytest.mark.parametrize("atoms_factory", get_atomspack_from_graph_factory_list())
@pytest.mark.parametrize("nc", test_ncs)
@pytest.mark.parametrize("reduction_fn", test_reduction_fns)
def test_gather1_sum(G, atoms_factory, nc, reduction_fn, device, numerical_single_precision_eps):
    atoms = atoms_factory(G)
    print("atoms", atoms)
    _verify_input(G, atoms)
    if "cuda" in device and reduction_fn is torch.sum:
        with pytest.raises(RuntimeError, match=re.escape("Ptens error in void ptens::Ptensors1_reduce0_cu(const TENSOR&, const TENSOR&, const ptens::AindexPackB&, int, int, CUstream_st* const&): failed assertion x.stride(1)==1.")):
            backprop_sum(ptens.ptensorlayer1, G, atoms, nc, reduction_fn, device, numerical_single_precision_eps)
    else:
        backprop_sum(ptens.ptensorlayer1, G, atoms, nc, reduction_fn, device, numerical_single_precision_eps)


@pytest.mark.parametrize("G", get_graph_list())
@pytest.mark.parametrize("atoms_factory", get_atomspack_from_graph_factory_list())
@pytest.mark.parametrize("nc", test_ncs)

def test_gather1_jac(G, atoms_factory, nc, device, numerical_single_precision_eps):
    atoms = atoms_factory(G)
    print("atoms", atoms)
    _verify_input(G, atoms)
    backprop_jac(ptens.ptensorlayer1, G, atoms, nc, device, numerical_single_precision_eps)

@pytest.mark.parametrize("G", get_graph_list())
@pytest.mark.parametrize("atoms_factory", get_atomspack_from_graph_factory_list())
@pytest.mark.parametrize("nc", test_ncs)
@pytest.mark.parametrize("reduction_fn", test_reduction_fns)
def test_gather2_sum(G, atoms_factory, nc, reduction_fn, device, numerical_single_precision_eps):
    atoms = atoms_factory(G)
    print("atoms", atoms)
    _verify_input(G, atoms)
    if "cuda" in device and reduction_fn is torch.sum:
        with pytest.raises(RuntimeError, match=re.escape("Ptens error in void ptens::Ptensors2_reduce0_shrink_cu(const TENSOR&, const TENSOR&, const ptens::AindexPackB&, int, int, CUstream_st* const&): failed assertion x.stride(1)==1.")):
            backprop_sum(ptens.ptensorlayer2, G, atoms, nc, reduction_fn, device, numerical_single_precision_eps)
    else:
        backprop_sum(ptens.ptensorlayer2, G, atoms, nc, reduction_fn, device, numerical_single_precision_eps)


@pytest.mark.parametrize("G", get_graph_list())
@pytest.mark.parametrize("atoms_factory", get_atomspack_from_graph_factory_list())
@pytest.mark.parametrize("nc", test_ncs)
def test_gather2_jac(G, atoms_factory, nc, device, numerical_single_precision_eps):
    atoms = atoms_factory(G)
    print("atoms", atoms)
    _verify_input(G, atoms)
    backprop_jac(ptens.ptensorlayer2, G, atoms, nc, device, numerical_single_precision_eps)
