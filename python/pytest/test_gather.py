import torch
import ptens
import pytest
import ptens_base
from conftest import numerical_grad_sum, numerical_jacobian, get_atomspack_from_graph_factory_list, get_graph_list

from torch.autograd.gradcheck import gradcheck


def get_graph_atomspack_nc():
    ncs = [2]
    graphs = get_graph_list()
    atomspack_factory = get_atomspack_from_graph_factory_list()

    test_list = []

    for g in graphs:
        for atompack in [factory(g) for factory in atomspack_factory]:
            for nc in ncs:
                test_list.append((g, atompack, nc))
    return test_list

def backprop_sum(cls, G, atoms, nc, device, numerical_single_precision_eps):
    x = cls.sequential(atoms, nc).to(device) + 1
    x = x.to(device)
    x.requires_grad_()
    print("x", x)
    atoms2 = G.subgraphs(ptens.subgraph.trivial())

    z = cls.gather(atoms2, x)

    zero = torch.zeros_like(z)
    l1_sum = torch.nn.L1Loss(reduction="sum")


    def loss_fn(x):
        z = cls.gather(atoms2, x)
        s = l1_sum(z, zero)
        # s = torch.sum(z)
        return s

    assert gradcheck(loss_fn, (x,), eps=numerical_single_precision_eps, rtol=1e-2, atol=1e-1, nondet_tol=1e-6)

    xgrad = torch.autograd.grad(outputs=loss_fn(x), inputs=x)[0]
    print("xgrad", xgrad)

    fn = lambda x: cls.gather(atoms2, x)
    xgrad2 = numerical_grad_sum(fn, x, numerical_single_precision_eps)
    print("xgrad2", xgrad2)
    assert torch.allclose(torch.Tensor(xgrad), torch.Tensor(xgrad2), rtol=1e-1, atol=1e-1)


    assert device in str(xgrad.device)

def backprop_jac(cls, G, atoms, nc, device, numerical_single_precision_eps):
    x = cls.sequential(atoms, nc).to(device)
    x.requires_grad_()
    x = x.to(device)
    atoms2 = G.subgraphs(ptens.subgraph.trivial())

    if cls.gather(atoms2, x).numel() > 0:

        def loss_fn(x):
            z = cls.gather(atoms2, x)
            return z

        xjac2 = numerical_jacobian(loss_fn, x, numerical_single_precision_eps)
        print("xjac2", xjac2)

        xjac = torch.autograd.functional.jacobian(loss_fn, x)
        print("xjac", xjac)
        xjac2 = xjac2.to(device)
        assert torch.allclose(xjac, xjac2, rtol=1e-1, atol=1e-1)
        assert gradcheck(loss_fn, (x,), eps=numerical_single_precision_eps, rtol=0.1, atol=0.1, nondet_tol=1e-6)

        assert device in str(xjac.device)


def _verify_input(G, atoms):
    N = max(G.adjacency_matrix().size())
    for i in range(len(atoms)):
        a = atoms[i]
        if len(a) > 0:
            assert max(a) < N

@pytest.mark.parametrize(("G", "atoms", "nc"), get_graph_atomspack_nc())
def test_gather0_sum(G, atoms, nc, device, numerical_single_precision_eps):
    _verify_input(G, atoms)
    backprop_sum(ptens.ptensorlayer0, G, atoms, nc, device, numerical_single_precision_eps)

@pytest.mark.parametrize(("G", "atoms", "nc"), get_graph_atomspack_nc())
def test_gather0_jac(G, atoms, nc, device, numerical_single_precision_eps):
    _verify_input(G, atoms)
    backprop_jac(ptens.ptensorlayer0, G, atoms, nc, device, numerical_single_precision_eps)

@pytest.mark.parametrize(('G', 'atoms', 'nc'), get_graph_atomspack_nc())
def test_gather1_sum(G, atoms, nc, device, numerical_single_precision_eps):
    _verify_input(G, atoms)
    backprop_sum(ptens.ptensorlayer1, G, atoms, nc, device, numerical_single_precision_eps)

@pytest.mark.parametrize(('G', 'atoms', 'nc'), get_graph_atomspack_nc())
def test_gather1_jac(G, atoms, nc, device, numerical_single_precision_eps):
    _verify_input(G, atoms)
    backprop_jac(ptens.ptensorlayer1, G, atoms, nc, device, numerical_single_precision_eps)

# @pytest.mark.parametrize(('G', 'atoms', 'nc'), graph_atoms_list)
# def test_gather2_sum(G, atoms, nc, device, numerical_single_precision_eps):
#     _verify_input(G, atoms)
#     backprop_sum(ptens.ptensorlayer2, G, atoms, nc, device, numerical_single_precision_eps)

# @pytest.mark.parametrize(('G', 'atoms', 'nc'), graph_atoms_list)
# def test_gather2_jac(G, atoms, nc, device, numerical_single_precision_eps):
#     _verify_input(G, atoms)
#     backprop_jac(ptens.ptensorlayer2, G, atoms, nc, device, numerical_single_precision_eps)
