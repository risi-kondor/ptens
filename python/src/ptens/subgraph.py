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
from ptens_base import subgraph as _subgraph


class subgraph:

    @classmethod
    def from_edge_index(self,M,n=-1,labels=None):
        G=subgraph()
        if labels is None:
            G.obj=_subgraph.edge_index(M,n)
        else:
            G.obj=_subgraph.edge_index(M,labels,n)
        return G

    @classmethod
    def from_matrix(self,M,labels=None):
        G=subgraph()
        if labels is None:
            G.obj=_subgraph.matrix(M)
        else:
            G.obj=_subgraph.matrix(M,labels)
        return G

    @classmethod
    def trivial(self):
        G=subgraph()
        G.obj=_subgraph.trivial()
        return G;

    @classmethod
    def edge(self):
        G=subgraph()
        G.obj=_subgraph.edge()
        return G;

    @classmethod
    def triangle(self):
        G=subgraph()
        G.obj=_subgraph.triangle()
        return G;


    def torch(self):
        return self.obj.dense()

    def __str__(self):
        return self.obj.__str__()

    def __repr__(self):
        return self.obj.__str__()

    
