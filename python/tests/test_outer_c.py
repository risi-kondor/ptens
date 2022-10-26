import torch
import ptens as p
import pytest

class TestOuter(object):
    
    def backprop(self, pts_1, pts_2, fn, _atoms, _nc):
        x = pts_1.randn(_atoms, _nc)
        y = pts_2.randn(_atoms, _nc)
        x.requires_grad_()
        y.requires_grad_()
        z=fn(x,y).to('cuda')

        testvec=z.randn_like().to('cuda')
        loss=z.inp(testvec).to('cuda')
        loss.backward(torch.tensor(1.0))
        xgrad=x.get_grad() 

        xeps = x.randn_like(0.01).to('cuda')
        yeps = y.randn_like(0.01).to('cuda')
        z = fn(x+xeps,y+yeps).to('cuda')
        xloss = z.inp(testvec).to('cuda')
        assert(torch.allclose(xloss-loss,xeps.inp(xgrad),rtol=1e-3, atol=1e-4))


    @pytest.mark.parametrize('nc', [1, 2, 4])
    @pytest.mark.parametrize('atoms', [[[1],[2],[6]],[[1],[2,5],[1,2,6]]])
    @pytest.mark.parametrize('pts_1', [p.ptensors0])
    @pytest.mark.parametrize('pts_2', [p.ptensors0, p.ptensors1, p.ptensors2])
    def test_outer0(self, pts_1, pts_2, nc, atoms):
        self.backprop(pts_1, pts_2, p.outer, atoms, nc)

    @pytest.mark.parametrize('nc', [1, 2, 4])
    @pytest.mark.parametrize('atoms', [[[1],[2],[6]],[[1],[2,5],[1,2,6]]])
    @pytest.mark.parametrize('pts_1', [p.ptensors1])
    @pytest.mark.parametrize('pts_2', [p.ptensors0, p.ptensors1])
    def test_outer1(self, pts_1, pts_2, nc, atoms):
        self.backprop(pts_1, pts_2, p.outer, atoms, nc)

    @pytest.mark.parametrize('nc', [1, 2, 4])
    @pytest.mark.parametrize('atoms', [[[1],[2],[6]],[[1],[2,5],[1,2,6]]])
    @pytest.mark.parametrize('pts_1', [p.ptensors2])
    @pytest.mark.parametrize('pts_2', [p.ptensors0])
    def test_outer2(self, pts_1, pts_2, nc, atoms):
        self.backprop(pts_1, pts_2, p.outer, atoms, nc)


