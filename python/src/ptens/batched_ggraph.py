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
from ptens_base import batched_ggraph as _batched_ggraph


class batched_ggraph:

    @classmethod
    def from_cache(self,keys):
        G=batched_ggraph()
        G.obj=_batched_ggraph(key)
        return G

    @classmethod
    def from_graphs(self,graphs):
        G=batched_ggraph()
        G.obj=_batched_ggraph([x.obj for x in graphs])
        return G

    @classmethod
    def from_edge_index(self,M,indicators):
        G=batched_ggraph()
        G.obj=_batched_ggraph(M,indicators)
        return G


    def __len__(self):
        return len(self.obj)
    
    def __getitem__(self,i):
        r=ptens.ggraph()
        r.obj=__getitem__(self.obj)
        return r

    
    def subgraphs(self,H):
        return self.obj.subgraphs(H.obj)

    def __str__(self):
        return self.obj.__str__()

    def __repr__(self):
        return self.obj.__str__()

    
