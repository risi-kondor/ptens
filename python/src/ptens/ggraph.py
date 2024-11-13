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
import ptens_base as pb
import ptens as p

from ptens_base import ggraph as _ggraph
from .exception import GraphCreationError

class ggraph:

    @classmethod
    def from_cache(self,key):
        G=ggraph()
        G.obj=_ggraph.from_cache(key)
        return G

    @classmethod
    def from_edge_index(cls,M,n=-1,labels=None):
        number_self_edges = M.shape[1] - torch.count_nonzero(M[0] - M[1])
        if number_self_edges > 0:
            raise GraphCreationError(f"Attempt to create a graph with self edges with the following edge index {M}")

        G=cls()
        G.obj=_ggraph.from_edge_index(M,n)
        if labels is not None:
            G.set_labels(labels)
        return G

    @classmethod
    def from_matrix(cls,M,labels=None):
        number_self_edges = torch.count_nonzero(torch.diag(M))
        if number_self_edges > 0:
            raise GraphCreationError(f"Attempt to create a graph with self edges with the following adjacency matrix index {M}")

        G=cls()
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


    def adjacency_matrix(self):
        return self.obj.dense()

    def is_labeled(self):
        return self.obj.is_labeled()
    
    def set_labels(self,labels):
        self.obj.set_labels(labels)
    
    def labels(self):
        return self.obj.get_labels()
    
    def subgraphs(self,H):
        return self.obj.subgraphs(H.obj)

    def csubgraphs(self,H,nvecs):
        H.set_evecs()
        return self.obj.csubgraphs(H.obj,nvecs)

    def cached_subgraph_lists(self):
        dict=self.obj.cached_subgraph_lists_as_map()
        r={}
        for a in dict:
            r[p.subgraph.make(a)]=dict[a]
        return r
    

    # ---- Caching -------------------------------------------------------------------------------------------


    def cache(self,key):
        self.obj.cache(key)


    # ---- I/O -----------------------------------------------------------------------------------------------

        
    def __str__(self):
        return self.obj.__str__()

    def __repr__(self):
        return self.obj.__str__()

    def __eq__(self, other):
        if self is other:
            return True
        if self.obj is other.obj:
            return True
        return self.obj.__eq__(other.obj)
