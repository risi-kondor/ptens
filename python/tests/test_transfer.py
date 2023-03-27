import torch
import ptens as p
import  pytest
class TestTransfer(object):
    def backprop(self,ptensorsk,_atoms,_nc, new_atoms):
        x=ptensorsk.randn(_atoms,_nc, _device=None)
        x.requires_grad_()
        z=p.transfer0(x,new_atoms)
        
        testvec=z.randn_like()
        loss=z.inp(testvec)
        loss.backward(torch.tensor(1.0))
        xgrad=x.get_grad()

        xeps=ptensorsk.randn(_atoms,_nc, _device=None)
        z=p.transfer0(x+xeps,new_atoms)
        xloss=z.inp(testvec)
        assert(torch.allclose(xloss-loss,xeps.inp(xgrad),rtol=1e-3, atol=1e-4))

   # @pytest.mark.parametrize('ptensorsk', [p.ptensors0, p.ptensors1, p.ptensors2])
   # @pytest.mark.parametrize('fn', [p.transfer0, p.transfer1, p.transfer2])
    @pytest.mark.parametrize('atoms', [[[1,2,3],[3,5],[2]]])
  #  @pytest.mark.parametrize('nc', [1, 2, 4])
    @pytest.mark.parametrize('new_atoms', [[[1,2],[2,5,1,2,6]]])
    def test_transfer0(self,atoms,new_atoms):
        self.backprop(p.ptensors0,atoms,2,new_atoms)


A = torch.tensor([[0,1,0],
                 [1,0,1],
                 [0,1,0]],dtype = torch.float32)
G=p.graph.from_matrix(A)
#print('G=',G)
#print(G.torch().dtype)

A1=torch.tensor([[0,1,1,2],
                 [1,0,2,1]],dtype=torch.float32)
G1=p.graph.from_edge_index(A1)
#print('G1=',G1)

G2 = p.graph.random(8,0.3) # (num of nodes, the prob of that there exists an edge between any two nodes)
#print('G2=', G2) 
#print(G2.torch())

# neighbors
G3 = p.graph.random(8,0.2)
print("G3's adges:", G3.edges())
print("=========================================================") 
print(G3.nhoods(0))
print(G3.nhoods(1))
print(G3.nhoods(2))
print(G3.nhoods(3))
print(G3.nhoods(4))
print(G3.nhoods(5)) # n
print(G3.nhoods(6)) # n 
print(G3.nhoods(7)) # n
print(G3.nhoods(8)) # n 
print(G3.nhoods(9)) # n


E=G3.edges()
#print("Edges:", E)

tri_Adjmatrix = torch.tensor([[0,1,1],[1,0,1],[1,1,0]],dtype = torch.float32)
#print("triangle adj =", tri_Adjmatrix)
tri_G = p.graph.from_matrix(tri_Adjmatrix)
#print("triangle=", tri_G)
#print("G3=", G3)
SubG = G3.subgraphs(tri_G)
#print("SubG=", SubG)
import ptens_base
A = ptens_base.atomspack([[0,1],[2],[4,5]])
B = ptens_base.atomspack([[1,3],[5,2],[0]])
G = p.graph.overlaps(A,B)
print("A=", A)
#print("overlaps:", G.torch())
