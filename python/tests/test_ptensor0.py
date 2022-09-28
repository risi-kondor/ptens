import torch
import ptens as p
import pytest

class TestPtensor0(object):
    def backprop(self, init_fn, fn, atoms, nc):
        p_x = init_fn.randn(atoms, nc)
        x = torch.tensor(p_x)
        x.requires_grad_()
        p_z = fn(p_x) 
        z = torch.tensor(p_z)
        
        atoms_z, nc_z = p_z.atoms, p_z.get_nc()
        p_pertur = init_fn.randn(atoms_z, nc_z)
        pertur = torch.tensor(p_pertur)
        pertur.requires_grad_()
        loss = z.dot(pertur)
        loss.backward(torch.tensor(1.0))
        xgrad = x.grad
        
        p_x_pertur = init_fn.randn(atoms, nc)
        x_pertur = torch.tensor(p_x_pertur)
        M = torch.tensor(p_x_pertur) + torch.tensor(p_x_pertur)
        p_M=p.ptensors0.from_matrix(M)
        p_z = fn(p_M)
        z = torch.tensor(p_z)
        xloss = z.dot(pertur)
        assert(torch.allclose(xloss - loss, x_pertur.dot(xgrad), rtol = 1e-3, atol = 1e-4))


    @pytest.mark.parametrize('nc', [1, 2, 4])
    def test_ptensor0_linmaps0(self, nc):
        self.backprop(p.ptensor0, p.linmaps0, [1,2,3], nc)

    @pytest.mark.parametrize('nc', [1, 2, 4])
    def test_ptensor0_linmaps1(self, nc):
        self.backprop(p.ptensor0, p.linmaps1, [1,2,3], nc)

    @pytest.mark.parametrize('nc', [1, 2, 4])
    def test_ptensor0_linmaps2(self, nc):
        self.backprop(p.ptensor0, p.linmaps2, [1,2,3], nc)

    @pytest.mark.parametrize('atoms', [[1], [2,3], [4,5,6]])
    def test_ptensor0_transfer0(self, atoms):
        self.backprop(p.ptensor0, p.transfer0, atoms, 3)

    @pytest.mark.parametrize('atoms', [[1], [2,3], [4,5,6]])
    def test_ptensor0_transfer1(self, atoms):
        self.backprop(p.ptensor0, p.transfer1, atoms, 3)

    @pytest.mark.parametrize('atoms', [[1], [2,3], [4,5,6]])
    def test_ptensor0_transfer2(self, atoms):
        self.backprop(p.ptensor0, p.transfer2, atoms, 3)
