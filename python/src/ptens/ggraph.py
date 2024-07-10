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
from ptens_base import ggraph as _ggraph


class ggraph:

    @classmethod
    def from_cache(self,key):
        G=ggraph()
        G.obj=_ggraph.from_cache(key)
        return G

    @classmethod
    def from_edge_index(self,M,n=-1,labels=None):
        G=ggraph()
        G.obj=_ggraph.from_edge_index(M,n)
        if labels is not None:
            G.set_labels(labels)
        return G

    @classmethod
    def from_matrix(self,M,labels=None):
        G=ggraph()
        G.obj=_ggraph.from_matrix(M)
        if labels is not None:
            G.set_labels(labels)
        return G

    @classmethod
    def random(self,_n,_p):
        G=ggraph()
        G.obj=_ggraph.random(_n,_p)
        return G


    # ---- Access --------------------------------------------------------------------------------------------


    def set_labels(self,labels):
        self.obj.set_labels(labels)
    
    def labels(self):
        return self.obj.get_labels()
    
    def torch(self):
        return self.obj.dense()

    def subgraphs(self,H):
        return self.obj.subgraphs(H.obj)


    # ---- Caching -------------------------------------------------------------------------------------------


    def cache(self,key):
        self.obj.cache(key)


    # ---- I/O -----------------------------------------------------------------------------------------------

        
    def __str__(self):
        return self.obj.__str__()

    def __repr__(self):
        return self.obj.__str__()

    
