import torch
import ptens as p
import pytest
class TestLinearMap(object):
    def backprop(self,ptensorsk,fn,_atoms,_nc):
        x=ptensorsk.randn(_atoms,_nc)
        x.requires_grad_()
        z=fn(x, normalized=False)
        
        testvec=z.randn_like()
        loss=z.inp(testvec)
        loss.backward(torch.tensor(1.0))
        xgrad=x.get_grad()

        xeps=ptensorsk.randn(_atoms,_nc)
        z=fn(x+xeps, normalized=False)
        xloss=z.inp(testvec)
        assert(torch.allclose(xloss-loss,xeps.inp(xgrad),rtol=1e-3, atol=1e-4))

    @pytest.mark.parametrize('ptensorsk', [p.ptensors0, p.ptensors1, p.ptensors2])
    @pytest.mark.parametrize('fn', [p.linmaps0, p.linmaps1, p.linmaps2])
    @pytest.mark.parametrize('atoms', [[1,2,3],[3,5],[2]])
    def test_linmapx(self,ptensorsk,fn,atoms):
        self.backprop(ptensorsk,fn,atoms,1)

    def test_linmapx(self,ptensorsk,fn,atoms):
        self.backprop(ptensorsk,fn,atoms,2)

    def test_linmapx(self,ptensorsk,fn,atoms):
        self.backprop(ptensorsk,fn,atoms,4)
