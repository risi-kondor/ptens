import torch
import ptens as p
import pytest

class TestReLU(object):

    def backprop(self,k, _alpha, _atoms, _nc):
        x = p.tlayer.randn(k,_atoms,_nc)
        x.requires_grad_()
        z=p.relu(x,_alpha)

        testvec=z.randn_like()
        loss=z.inp(testvec)
        print(1)
        loss.backward(torch.tensor(1.0))
        print(2)
        xgrad=x.get_grad()

        xeps = x.randn_like()
        z = p.relu(x+xeps,_alpha)
        print(3)
        xloss = z.inp(testvec)
        assert(torch.allclose(xloss-loss,xeps.inp(xgrad),rtol=1e-3, atol=1e-4))

    @pytest.mark.parametrize('k', [0,1,2])
    @pytest.mark.parametrize('alpha', [0.1, 0.2, 0.3])
    @pytest.mark.parametrize('nc', [3,8])
    @pytest.mark.parametrize('atoms', [[[1],[2],[6]],[[1],[2,3],[2,5,8]]])
    def test_relu(self, k, alpha, atoms, nc):
        self.backprop(k, alpha, atoms, nc)
