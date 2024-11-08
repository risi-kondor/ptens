import torch
import ptens


def test_create_tensor(device):
    A = torch.ones((3, 5))
    sum_a = float(torch.sum(A**2))
    assert A.shape == (3, 5)
    B = A.to(device)
    sum_b = float(torch.sum(B**2))
    assert device in str(B.device)
    assert abs(sum_a - sum_b) < 1e-6


def test_create_ptensor0(device, float_epsilon):
    A = ptens.ptensor0.randn([2], 5)
    sum_a = float(torch.sum(A**2))
    B = A.to(device)
    sum_b = float(torch.sum(B**2))
    print(B)
    assert B.atoms == [2]
    assert B.shape == (5,)
    assert device in str(B.device)
    assert sum_a > 1e-3
    assert abs(sum_a - sum_b) < float_epsilon


def test_create_ptensor1(device, float_epsilon):
    A = ptens.ptensor1.randn([1, 2, 3], 5)
    sum_a = float(torch.sum(A**2))
    B = A.to(device)
    sum_b = float(torch.sum(B**2))
    print(B)
    assert B.atoms == [1, 2, 3]
    assert B.shape == (3, 5)
    assert device in str(B.device)
    assert abs(sum_a - sum_b) < float_epsilon
    assert sum_a > 1e-3


def test_create_ptensor2(device, float_epsilon):
    A = ptens.ptensor2.randn([1, 2, 3], 5)
    sum_a = float(torch.sum(A**2))
    B = A.to(device)
    sum_b = float(torch.sum(B**2))
    print(B)
    assert B.atoms == [1, 2, 3]
    assert B.shape == (3, 3, 5)
    assert device in str(B.device)
    assert abs(sum_a - sum_b) < float_epsilon
    assert sum_a > 1e-3


def test_sequential(device, float_epsilon):
    A = ptens.ptensor1.sequential([1, 2, 3], 5)
    sum_a = float(torch.sum(A))
    B = A.to(device)
    sum_b = float(torch.sum(B))

    assert abs(sum_a - sum_b) < float_epsilon
    assert abs(sum_a - sum(range(3 * 5))) < float_epsilon
