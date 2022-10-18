import torch
import ptens as p
import ptens_base
import pytest

class TestReLU(object):

    def backprop(self,pts, fn, _alpha, _atoms, _nc):
        x = pts.randn(_atoms, _nc)
        x.requires_grad_()
        z=fn(x,_alpha)

        testvec=z.randn_like()
        loss=z.inp(testvec)
        loss.backward(torch.tensor(1.0))
        xgrad=x.get_grad()

        xeps = x.randn_like(0.01)
        z = fn(x+xeps,_alpha)
        xloss = z.inp(testvec)
        assert(torch.allclose(xloss-loss,xeps.inp(xgrad),rtol=1e-3, atol=1e-4))

    @pytest.mark.parametrize('alpha', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    @pytest.mark.parametrize('nc', [1,2,4,5,6,8,9])
    @pytest.mark.parametrize('atoms', [[[1],[2],[6]],[[1],[2,3],[2,5,8]]])
    def test_relu0(self, alpha, atoms, nc):
        self.backprop(p.ptensors0, p.relu, alpha, atoms, nc)

    @pytest.mark.parametrize('alpha', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    @pytest.mark.parametrize('nc', [1,2,4,5,6,8,9])
    @pytest.mark.parametrize('atoms', [[[1],[2],[6]],[[1],[2,3],[2,5,8]]])
    def test_relu1(self, alpha, atoms, nc):
        self.backprop(p.ptensors1, p.relu, alpha, atoms, nc)

    @pytest.mark.parametrize('alpha', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    @pytest.mark.parametrize('nc', [1,2,4,5,6,8,9])
    @pytest.mark.parametrize('atoms', [[[1],[2],[6]],[[1],[2,3],[2,5,8]]])
    def test_relu2(self, alpha, atoms, nc):
        self.backprop(p.ptensors2, p.relu, alpha, atoms, nc)
