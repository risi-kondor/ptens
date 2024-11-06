import os

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
        import torch

        assert torch.cuda.is_available()

    return device


@pytest.fixture(scope="session")
def float_epsilon():
    return 1e-5
