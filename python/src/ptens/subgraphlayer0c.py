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

import ptens as p
import ptens_base as pb 


#import ptens.ptensorsc_base as ptensorsc_base
#import ptens.ptensor0c as ptensor0c
import ptens.ptensorlayer0c as ptensorlayer0c


class subgraphlayer0c(ptensorlayer0c):

    def __init__(self,graph,subgraph,x):
       assert isinstance(graph,p.ggraph)
       assert isinstance(subgraph,p.subgraph)
       assert isinstance(x,p.ptensorlayer0c)
       super(ptensorlayer0c,self).__init__(x)
       graph=graph
       subgraph=subgraph

    @classmethod
    def zeros(self,graph,subgraph,nc,device='cpu'):
        assert isinstance(graph,p.ggraph)
        assert isinstance(subgraph,p.subgraph)
        assert isinstance(nc,int)
        atoms=graph.subgraphs(subgraph)
        R=subgraphlayer0c(torch.zeros([atoms.nrows0(),nc],device=device))
        R.graph=graph
        R.subgraph=subgraph
        R.atoms=atoms
        return R

    @classmethod
    def randn(self,graph,subgraph,nc,device='cpu'):
        assert isinstance(graph,p.ggraph)
        assert isinstance(subgraph,p.subgraph)
        assert isinstance(nc,int)
        atoms=graph.subgraphs(subgraph)
        R=subgraphlayer0c(torch.randn([atoms.nrows0(),nc],device=device))
        R.graph=graph
        R.subgraph=subgraph
        R.atoms=atoms
        return R

    @classmethod
    def from_matrix(self,graph,subgraph,M):
        assert isinstance(graph,p.ggraph)
        assert isinstance(subgraph,p.subgraph)
        assert isinstance(M,torch.Tensor)
        assert M.dim()==2
        atoms=graph.subgraphs(subgraph)
        assert M.size(0)==atoms.tsize0()
        R=subgraphlayer0c(M)
        R.graph=graph
        R.subgraph=subgraph
        R.atoms=atoms
        return R

    def clone(self):
        r=subgraphlayer0c(super().clone())
        r.atoms=self.atoms
        r.graph=self.graph
        r.subgraph=self.subgraph
        return r

    
    # ---- Linmaps ------------------------------------------------------------------------------------------


    @classmethod
    def linmaps(self,x):
        assert isinstance(x,subgraphlayer0c) or isinstance(x,subgraphlayer1c)
        return subgraphlayer0c(x.graph,x.subgraph,ptensorlayer0c.linmaps(x))


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return "subgraphlayer0c(len="+str(self.size(0))+",nc="+str(self.size(1))+")"

    def __str__(self):
        r=""
        for i in range(len(self)):
            r=r+str(self[i])+"\n\n"
        return r













    # ---- Operations ----------------------------------------------------------------------------------------


#     def __add__(self,y):
#         assert self.size==y.size
#         assert self.atoms==y.atoms
#         r=self.clone()
#         r+=y
#         return r


