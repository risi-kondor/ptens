import os
import torch
import pytest


@pytest.fixture(scope="session")
def ptens_cuda_support():
    import ptens_base

    string = ptens_base.status_str().split("\n")
    for line in string:
        if "CUDA support" in line:
            if "ON" in line:
                return True
            if "OFF" in line:
                return False
    assert False


@pytest.fixture(scope="session")
def device(ptens_cuda_support):
    device = os.environ["TORCH_TEST_DEVICE"]

    if "cuda" in device:
        assert ptens_cuda_support
        assert torch.cuda.is_available()

    return device


@pytest.fixture(scope="session")
def float_epsilon():
    return 1e-5


def numerical_grad_sum(fn, x, h):
    grad = torch.zeros_like(x)
    for i in range(x.numel()):
        xp = x.clone()
        xp.view(-1)[i] += h
        xm = x.clone()
        xm.view(-1)[i] -= h

        # Using torch.sum here, because torch autograd, calcualtes the partial diff of a scalar valued functino.
        # With sum, we can a scalar valued function, and the summed parts factorize
        num_diff = torch.sum(fn(xp)) - torch.sum(fn(xm))
        grad_value = num_diff / (2 * float(h))        
        grad.view(-1)[i] = grad_value
    return grad

@pytest.mark.parametrize("m,c", [(0., 3.), (0.5, -0.3), (-0.8, 0.2)])
def test_numerical_grad_linear(m, c):
    def linear(x):
        return m*x + c

    x = torch.randn((5,10))
    grad = numerical_grad_sum(linear, x, 1e-2)
    ana_grad = torch.ones_like(x) * m

    allclose = torch.allclose(ana_grad, grad, rtol=1e-3, atol=1e-5)
    if not allclose:
        print(f"Max absolute difference: {torch.max(torch.abs(ana_grad - grad))}")
        print(f"Mean absolute difference: {torch.mean(torch.abs(ana_grad - grad))}")
        print(f"Numerical grad range: [{grad.min()}, {grad.max()}]")
        print(f"Analytical grad range: [{ana_grad.min()}, {ana_grad.max()}]")
    
    assert allclose

@pytest.mark.parametrize("a,b,c", [(1. ,2., 3.), (-0.5, 0.4, -0.3), (1.2, -0.8, 0.2)])
def test_numerical_grad_square(a, b, c):
    from torch.autograd.gradcheck import gradcheck
    def square(x):
        return a*x**2 + b*x + c

    x = torch.randn((5,10))
    grad = numerical_grad_sum(square, x, 1e-3)
    ana_grad = 2*a*x + b

    allclose = torch.allclose(ana_grad, grad, rtol=1e-2, atol=1e-2)

    if not allclose:
        print(f"Max absolute difference: {torch.max(torch.abs(ana_grad - grad))}")
        print(f"Mean absolute difference: {torch.mean(torch.abs(ana_grad - grad))}")
        print(f"Numerical grad range: [{grad.min()}, {grad.max()}]")
        print(f"Analytical grad range: [{ana_grad.min()}, {ana_grad.max()}]")
    
    assert allclose
    x.requires_grad_()
    assert gradcheck(square, (x,), eps=1e-2, rtol=1e-2, atol=1e-2)
    
    
# Add a test against autograd for validation
def test_against_autograd():
    def complex_function(x):
        return torch.sum(torch.sin(x) + x**2)

    x = torch.randn(5, 10, requires_grad=True)
    
    # Compute gradient using autograd
    y = complex_function(x)
    y.backward()
    autograd_grad = x.grad

    # Compute gradient using numerical method
    numerical_grad = numerical_grad_sum(complex_function, x.detach(), 1e-3)

    allclose = torch.allclose(autograd_grad, numerical_grad, rtol=1e-2, atol=1e-2)
    if not allclose:
        print(f"Max absolute difference: {torch.max(torch.abs(autograd_grad - numerical_grad))}")
        print(f"Mean absolute difference: {torch.mean(torch.abs(autograd_grad - numerical_grad))}")
    

    assert allclose
