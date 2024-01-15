import torch
import ptens as p
import pytest

class TestPtensors1b(object):

    def backprop(self,fn,G,S,_nc):
        x=p.subgraphlayer1b.randn(G,S,_nc)
        x.requires_grad_()
        z=fn(x)
        
        testvec=z.randn_like()
        loss=z.inp(testvec)
        loss.backward(torch.tensor(1.0))
        xgrad=x.get_grad()

        xeps=p.subgraphlayer1b.randn(G,S,_nc)
        z=fn(x+xeps)
        xloss=z.inp(testvec)
        assert(torch.allclose(xloss-loss,xeps.inp(xgrad),rtol=1e-3, atol=1e-4))


    def backprop2(self,fn,_atoms1,_atoms2,_nc):
        x=p.ptensors1b.randn(_atoms1,_nc)
        x.requires_grad_()
        z=fn(x,_atoms2)
        
        testvec=z.randn_like()
        loss=z.inp(testvec)
        loss.backward(torch.tensor(1.0))
        xgrad=x.get_grad()

        xeps=p.ptensors1b.randn(_atoms1,_nc)
        z=fn(x+xeps,_atoms2)
        xloss=z.inp(testvec)
        assert(torch.allclose(xloss-loss,xeps.inp(xgrad),rtol=1e-3, atol=1e-4))


    @pytest.mark.parametrize('N', [5,10,15])
    @pytest.mark.parametrize('nc', [1, 2, 4])
    def test_linmaps0(self,N,nc):
        G=p.ggraph.random(N,0.5)
        S=p.subgraph.edge()
        self.backprop(p.subgraphlayer0b.linmaps,G,S,nc)

    @pytest.mark.parametrize('N', [5,10,15])
    @pytest.mark.parametrize('nc', [1, 2, 4])
    def test_linmaps1(self,N,nc):
        G=p.ggraph.random(N,0.5)
        S=p.subgraph.edge()
        self.backprop(p.subgraphlayer1b.linmaps,G,S,nc)

#    @pytest.mark.parametrize('N', [5,10,15])
#    @pytest.mark.parametrize('atoms', [[[1],[2],[6]],[[1],[2],[6,8,9]]])
#    @pytest.mark.parametrize('nc', [1, 2, 4])
#    def test_linmaps2(self,atoms,nc):
#        self.backprop(p.ptensors2b.linmaps,atoms,nc)

    @pytest.mark.parametrize('N', [5,10,15])
#    @pytest.mark.parametrize('atoms1', [[[1],[2],[6]],[[1],[2,5],[1,2,6]]])
#    @pytest.mark.parametrize('atoms2', [[[1,2],[4],[3,5]],[[2],[5],[1,3]]])
#    @pytest.mark.parametrize('nc', [1, 2, 4])
#    def test_linmaps0(self,atoms1,atoms2,nc):
#        self.backprop2(p.ptensors0b.gather,atoms1,atoms2,nc)

