import os
import torch
import pytest
import ptens
import ptens_base


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


@pytest.fixture(scope="session")
def numerical_single_precision_eps():
    return 1e-3


def get_graph_list():
    graph_list = [
        ptens.ggraph.from_edge_index(torch.Tensor([[], []]).int()), #Simplest graph
        ptens.ggraph.from_edge_index(torch.Tensor([[0, 1], [1, 0]]).int()), # Simple graph
        ptens.ggraph.from_edge_index(torch.Tensor( # Two unconnected rings
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
             [1, 2, 3, 4, 5, 0, 7, 8, 9, 6,]]).int()),
        ptens.ggraph.from_edge_index(torch.Tensor( # Two connected rings
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
             [1, 2, 3, 4, 5, 0, 7, 8, 9, 6, 9,]]).int()),
        ptens.ggraph.from_edge_index(torch.Tensor( # star
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 2, 3, 4, 5, 7, 8, 9, 6,]]).int()),
        ptens.ggraph.from_edge_index(torch.Tensor([[0, 1], [1, 0]]).int(), labels=torch.Tensor([2, 4]).int()), # Simple graph with labels
        ptens.ggraph.from_matrix(torch.Tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]).int()), # From Matrix
        ptens.ggraph.from_matrix(torch.Tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]).int(), labels=torch.Tensor([4, 5, 6]).int()), # From Matrix
        ptens.ggraph.random(0, 0.5),
        ptens.ggraph.random(0, 0.),
        ptens.ggraph.random(0, 1.0),
        ptens.ggraph.random(10, 0.0),
        ptens.ggraph.random(10, 1.0),
        ptens.ggraph.random(10, 0.5),
        ptens.ggraph.random(2, 1.0),
        ptens.ggraph.random(9, 1.0),

    ]

    return graph_list


def get_atomspack_from_graph_factory_list():
    def range_atomspack(g):
        N = g.adjacency_matrix().size()[0]
        return ptens_base.atomspack.from_list([[i] for i in range(N)])

    def empty(g):
        return ptens_base.atomspack.from_list([[]])

    def empty2(g):
        return ptens_base.atomspack.from_list([[]])

    def trivial(g):
        return g.subgraphs(ptens.subgraph.trivial())

    def edge(g):
        return g.subgraphs(ptens.subgraph.edge())

    def triangle(g):
        return g.subgraphs(ptens.subgraph.triangle())

    def cycle(g, n):
        return g.subgraphs(ptens.subgraph.cycle(n))

    def star(g, n):
        return g.subgraphs(ptens.subgraph.star(n))

    factory_list = [
        range_atomspack,
        empty,
        empty2,
        trivial,
        edge,
        triangle,
        lambda g: cycle(g, 4),
        lambda g: star(g, 3),
        ]

    return factory_list


def numerical_grad_sum(fn, x, h):
    grad = torch.zeros_like(x)
    for i in range(x.numel()):
        xp = x.clone()
        xp.view(-1)[i] += h
        xm = x.clone()
        xm.view(-1)[i] -= h

        # Using torch.sum here, because torch autograd, calcualtes the partial diff of a scalar valued functino.
        # With sum, we can a scalar valued function, and the summed parts factorize
        a = fn(xp).double()
        b = fn(xm).double()
        num_diff = torch.sum(a)- torch.sum(b)
        grad_value = num_diff / (2 * float(h))
        grad.view(-1)[i] = grad_value
    return grad


def numerical_jacobian(fn, x, h=1e-5):
    # Get the shape of the input and output
    input_shape = x.shape
    output_shape = fn(x).shape

    # Flatten the input
    x_flat = x.view(-1)

    # Initialize the Jacobian matrix
    jacobian = torch.zeros(output_shape.numel(), input_shape.numel())

    for i in range(x_flat.numel()):
        xp = x.clone()
        xm = x.clone()
        xp.view(-1)[i] += h
        xm.view(-1)[i] -= h

        # Compute function values
        fp = fn(xp)
        fm = fn(xm)

        # Compute partial derivatives
        partial_deriv = (fp - fm) / (2 * h)

        # Flatten the partial derivatives and store in Jacobian
        jacobian[:, i] = partial_deriv.view(-1)

    # Reshape Jacobian to match input and output shapes
    jacobian = jacobian.view(output_shape + input_shape)

    return jacobian


@pytest.mark.parametrize("m,c", [(0.0, 3.0), (0.5, -0.3), (-0.8, 0.2)])
def test_numerical_grad_linear(m, c, numerical_single_precision_eps):
    def linear(x):
        return m * x + c

    x = torch.randn((5, 10))
    grad = numerical_grad_sum(linear, x, numerical_single_precision_eps)
    ana_grad = torch.ones_like(x) * m

    allclose = torch.allclose(ana_grad, grad, rtol=1e-3, atol=1e-5)
    if not allclose:
        print(f"Max absolute difference: {torch.max(torch.abs(ana_grad - grad))}")
        print(f"Mean absolute difference: {torch.mean(torch.abs(ana_grad - grad))}")
        print(f"Numerical grad range: [{grad.min()}, {grad.max()}]")
        print(f"Analytical grad range: [{ana_grad.min()}, {ana_grad.max()}]")

    assert allclose


@pytest.mark.parametrize("a,b,c", [(1.0, 2.0, 3.0), (-0.5, 0.4, -0.3), (1.2, -0.8, 0.2)])
def test_numerical_grad_square(a, b, c, numerical_single_precision_eps):
    from torch.autograd.gradcheck import gradcheck

    def square(x):
        return a * x**2 + b * x + c

    x = torch.randn((5, 10))
    grad = numerical_grad_sum(square, x, numerical_single_precision_eps)
    ana_grad = 2 * a * x + b

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
def test_numerical_grad_against_autograd(numerical_single_precision_eps):
    def complex_function(x):
        return torch.sum(torch.sin(x) + x**2)

    x = torch.randn(5, 10, requires_grad=True)

    # Compute gradient using autograd
    y = complex_function(x)
    y.backward()
    autograd_grad = x.grad

    # Compute gradient using numerical method
    numerical_grad = numerical_grad_sum(complex_function, x.detach(), numerical_single_precision_eps)

    allclose = torch.allclose(autograd_grad, numerical_grad, rtol=1e-2, atol=1e-2)
    if not allclose:
        print(f"Max absolute difference: {torch.max(torch.abs(autograd_grad - numerical_grad))}")
        print(f"Mean absolute difference: {torch.mean(torch.abs(autograd_grad - numerical_grad))}")

    assert allclose


@pytest.mark.parametrize("m,c", [(0.0, 3.0), (0.5, -0.3), (-0.8, 0.2)])
def test_numerical_jacobian_linear(m, c, numerical_single_precision_eps):
    def linear(x):
        return m * x + c

    x = torch.randn((5, 10))
    jac = numerical_jacobian(linear, x, numerical_single_precision_eps)
    ana_jac = torch.zeros_like(jac)

    for i in range(jac.size()[0]):
        for j in range(jac.size()[1]):
            ana_jac[i, j, i, j] = m

    # torch.autograd.functional.jacobian(linear, x)

    allclose = torch.allclose(ana_jac, jac, rtol=1e-3, atol=1e-5)
    if not allclose:
        print(f"Max absolute difference: {torch.max(torch.abs(ana_jac - jac))}")
        print(f"Mean absolute difference: {torch.mean(torch.abs(ana_jac - jac))}")
        print(f"Numerical grad range: [{jac.min()}, {jac.max()}]")
        print(f"Analytical grad range: [{ana_jac.min()}, {ana_jac.max()}]")

    assert allclose


@pytest.mark.parametrize("a,b,c", [(1.0, 2.0, 3.0), (-0.5, 0.4, -0.3), (1.2, -0.8, 0.2)])
def test_numerical_jacobian_square(a, b, c, numerical_single_precision_eps):
    def square(x):
        return a * x**2 + b * x + c

    x = torch.randn((5, 10))
    jac = numerical_jacobian(square, x, numerical_single_precision_eps)
    ana_jac = torch.zeros_like(jac)

    value = 2 * a * x + b
    for i in range(jac.size()[0]):
        for j in range(jac.size()[1]):
            ana_jac[i, j, i, j] = value[i, j]

    allclose = torch.allclose(ana_jac, jac, rtol=1e-2, atol=1e-2)

    if not allclose:
        print(f"Max absolute difference: {torch.max(torch.abs(ana_jac - jac))}")
        print(f"Mean absolute difference: {torch.mean(torch.abs(ana_jac - jac))}")
        print(f"Numerical jac range: [{jac.min()}, {jac.max()}]")
        print(f"Analytical jac range: [{ana_jac.min()}, {ana_jac.max()}]")

    assert allclose


# Add a test against autograd for validation


def test_numerical_jacobian_against_autograd(numerical_single_precision_eps):
    def complex_function(x):
        return torch.sin(x) + x**2

    x = torch.randn(5, 10, requires_grad=True)

    # Compute gradient using numerical method
    numerical_jac = numerical_jacobian(complex_function, x.detach(), numerical_single_precision_eps)

    autograd_jac = torch.autograd.functional.jacobian(complex_function, x)

    allclose = torch.allclose(autograd_jac, numerical_jac, rtol=1e-2, atol=1e-2)
    if not allclose:
        print(f"Max absolute difference: {torch.max(torch.abs(autograd_jac - numerical_jac))}")
        print(f"Mean absolute difference: {torch.mean(torch.abs(autograd_jac - numerical_jac))}")

    assert allclose
