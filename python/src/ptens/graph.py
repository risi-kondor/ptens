import torch
import ptens_base
from ptens_base import graph as _graph


class graph:

    @classmethod
    def from_matrix(self,M):
        G=graph()
        G.obj=_graph.matrix(M)
        return G

    @classmethod
    def random(self,_n,_p):
        G=graph()
        G.obj=_graph.random(_n,_p)
        return G

    def torch(self,):
        return self.obj.dense()

    def nhoods(self,_l):
        return self.obj.nhoods(_l)

    def __str__(self):
        return self.obj.__str__()

    def __repr__(self):
        return self.obj.__str__()

    
