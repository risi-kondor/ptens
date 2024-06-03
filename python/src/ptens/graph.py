#
# This file is part of ptens, a C++/CUDA library for permutation 
# equivariant message passing. 
#  
# Copyright (c) 2023, Imre Risi Kondor
#
# This source code file is subject to the terms of the noncommercial 
# license distributed with cnine in the file LICENSE.TXT. Commercial 
# use is prohibited. All redistributed versions of this file (in 
# original or modified form) must retain this copyright notice and 
# must be accompanied by a verbatim copy of the license. 
#
#
import torch
import ptens_base
from ptens_base import graph as _graph


class graph:

    @classmethod
    def from_edge_index(self,M,n=-1,labels=None,m=None):
        G=graph()
        if labels is None:
            if m is None:
                G.obj=_graph.edge_index(M,n)
            else:
                G.obj=_graph.edge_index(M,n,m)
        else:
            G.obj=_graph.edge_index(M,labels,n)
        return G

    @classmethod
    def from_matrix(self,M,labels=None):
        G=graph()
        if labels is None:
            G.obj=_graph.matrix(M)
        else:
            G.obj=_graph.matrix(M,labels)
        return G

    @classmethod
    def random(self,_n,_p):
        G=graph()
        G.obj=_graph.random(_n,_p)
        return G

    @classmethod
    def randomd(self,_n,_p):
        G=graph()
        G.obj=_graph.randomd(_n,_p)
        return G

    @classmethod
    def overlaps(self,x,y):
        G=graph()
        G.obj=_graph.overlaps(x,y)
        return G

    def torch(self):
        return self.obj.dense()

    def nhoods(self,_l):
        return self.obj.nhoods(_l)

    def edges(self):
        return self.obj.edges()

    #def subgraphs(self,H):
    #    return self.obj.subgraphs(H.obj)

    def __str__(self):
        return self.obj.__str__()

    def __repr__(self):
        return self.obj.__str__()

    
