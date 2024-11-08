import torch
import ptens_base as pb
import ptens as p


def test_ptensorlayer(device, float_epsilon):
    atoms = pb.atomspack.from_list([[1, 3, 4], [2, 5], [0, 2]])
    atoms2 = pb.atomspack.random(5, 5, 0.6)

    A0 = p.ptensorlayer0.randn(atoms, 3)
    A1 = p.ptensorlayer1.randn(atoms, 3)
    A2 = p.ptensorlayer2.randn(atoms, 3)

    B0 = p.ptensorlayer1.gather(atoms2, A0)
    B1 = p.ptensorlayer1.gather(atoms2, A1)
    B2 = p.ptensorlayer1.gather(atoms2, A2)

    A0g = A0.to(device)
    A1g = A1.to(device)
    A2g = A2.to(device)

    B0g = p.ptensorlayer1.gather(atoms2, A0g)
    B1g = p.ptensorlayer1.gather(atoms2, A1g)
    B2g = p.ptensorlayer1.gather(atoms2, A2g)

    # Making sure we don't just compute on zeros
    assert torch.norm(B0) > float_epsilon
    assert torch.norm(B1) > float_epsilon
    assert torch.norm(B2) > float_epsilon

    assert torch.allclose(B0, B0g.to("cpu"))
    assert torch.allclose(B1g.to("cpu"), B1)
    assert torch.allclose(B2g.to("cpu"), B2)
