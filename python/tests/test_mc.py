import torch
import ptens as p
import pytest
class TestMc(object):
    
    def backprop(self, pts, _atoms, _nc):
        x = pts.randn(_atoms, _nc)
        y = torch.randn(_nc,3) 
        x.requires_grad_()
        z=x.mult_channels(y)

        testvec=z.randn_like()
        loss=z.inp(testvec)
        loss.backward(torch.tensor(1.0))
        xgrad=x.get_grad()

        xeps = x.randn_like()
        x_ = x+xeps
        z = x_.mult_channels(y)
        xloss = z.inp(testvec)
        assert(torch.allclose(xloss-loss,xeps.inp(xgrad),rtol=1e-3, atol=1e-4))


    @pytest.mark.parametrize('nc', [1, 2, 4])
    @pytest.mark.parametrize('atoms', [[[1],[2],[6]],[[1],[2,5],[1,2,6]]])
    def test_mc0(self, nc, atoms):
        self.backprop(p.ptensors0, atoms, nc)

    @pytest.mark.parametrize('nc', [1, 2, 4])
    @pytest.mark.parametrize('atoms', [[[1],[2]],[[2,5],[1,2,6]]])
    def test_mc1(self, nc, atoms):
        self.backprop(p.ptensors1, atoms, nc)

    @pytest.mark.parametrize('nc', [1, 2, 4])
    @pytest.mark.parametrize('atoms', [[[1],[2],[6]],[[1],[2,5],[1,2,6]]])
    def test_mc2(self, nc, atoms):
        self.backprop(p.ptensors2, atoms, nc)

