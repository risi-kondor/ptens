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
from ptens_base import batched_subgraphlayer1b as _batched_subgraphlayer1b
from ptens.utility import device_id as device_id
from ptens.utility import device_str as device_str
from ptens.ptensorsb import * 
#from ptens.batched_ptensors1b import BatchedPtensors1b_LinmapsFn
from ptens.batched_ptensors1b import BatchedPtensors1b_GatherFn


class batched_subgraphlayer1b(torch.Tensor):

    @classmethod
    def dummy(self):
        R=batched_subgraphlayer1b(1)
        return R

    @classmethod
    def init(self,obj):
        R=batched_subgraphlayer1b(1)
        R.obj=obj
        return R
    
#    @classmethod
#    def from_matrix(self,M,atoms):
#        return Batched_subgraphlayer1b_fromMxFn.apply(M,atoms)
            
    @classmethod
    def like(self,x,M):
        return Batched_subgraphlayer1b_likeFn.apply(x,M)

#     @classmethod
#     def zeros(self, _G, _nc, device='cpu'):
#         R=batched_subgraphlayer1b(1)
#         R.obj=_batched_subgraphlayer1b.create(_G.obj,_nc,0,device_id(device))
#         return R

#     @classmethod
#     def randn(self, _G, _nc, device='cpu'):
#         R=batched_subgraphlayer1b(1)
#         R.obj=_batched_subgraphlayer1b.create(_G.obj,_nc,4,device_id(device))
#         return R

#     @classmethod
#     def sequential(self, _G, _nc, device='cpu'):
#         R=batched_subgraphlayer1b(1)
#         R.obj=_batched_subgraphlayer1b.create(_G.obj,_nc,3,device_id(device))
#         return R

    def randn_like(self):
        return batched_subgraphlayer1b.init(self.obj.randn_like())
    
    @classmethod
    def cat(self,*args):
        return Batched_subgraphlayer1b_catFn.apply(self,*args)


    # ----- Access -------------------------------------------------------------------------------------------


    def _get_grad(self):
        return self.obj.get_grad()

    def get_grad(self):
        R=batched_subgraphlayer1b(1)
        R.obj=self.obj.get_grad()
        return R
    
    def get_dev(self):
        return self.obj.get_dev()

    def get_nc(self):
        return self.obj.get_nc()

    def get_atoms(self):
        return self.obj.get_atoms()
    
    def atoms_of(self, i):
        return self.obj.atoms_of(i)

    def torch(self):
        return Ptensorsb_toMxFn.apply(self)
        
    def to(self, device='cpu'):
        return Ptensorsb_toFn.apply(self,device)


    # ---- Operations ----------------------------------------------------------------------------------------


    def __add__(self,y):
        return Ptensorsb_addFn.apply(self,y)

    def cat_channels(self,y):
        return Ptensorsb_cat_channelsFn.apply(self,y)

    def outer(self,y):
         return Batched_subgraphlayer1b_outerFn.apply(self,y)

    def __mul__(self,y):
        return Ptensorsb_mprodFn.apply(self,y)

    def linear(self,y,b):
        return Ptensorsb_linearFn.apply(self,y,b)

    def scale(self,y):
        return Ptensorsb_scaleFn.apply(self,y)

    def avg_pool(self):
        return Ptensorsb_averageFn.apply(self)

    def mult_channels(self,y):
        return Ptensorsb_mult_channelsFn.apply(self,y)

    def relu(self,alpha=0.1):
        return Ptensorsb_ReLUFn.apply(self,alpha)
        
    def inp(self,y):
        return Ptensorsb_inpFn.apply(self,y)
    
    def diff2(self,y):
        return Ptensorsb_diff2Fn.apply(self,y)


    # ---- Message passing -----------------------------------------------------------------------------------
    

    @classmethod
    def linmaps(self,x):
        return BatchedSubgraphlayer1b_LinmapsFn.apply(x)

    @classmethod
    def gather(self,x,S,min_overlaps=1):
        return BatchedSubgraphlayer1b_GatherFn.apply(x,S,min_overlaps)

    @classmethod
    def gather_from_ptensors(self,x,G,S,min_overlaps=1):
        return BatchedSubgraphlayer1b_GatherFromPtensorsbFn.apply(x,G,S,min_overlaps)

    def autobahn(self,w,b):
        return BatchedSubgraphlayer1b_autobahnFn.apply(self,w,b)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------


class BatchedSubgraphlayer1b_fromEdgeFeaturesFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,graphs,M):
        R=batched_subgraphlayer1b.dummy()
        R.obj=_batched_subgraphlayer1b.from_edge_features(graphs,M)
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        return None, ctx.r.get_grad().torch()


class Batched_subgraphlayer1b_likeFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,M):
        r=batched_subgraphlayer1b(1)
        r.obj=_batched_subgraphlayer1b.like(x.obj,M)
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        return None, ctx.r.get_grad().torch()


class Batched_subgraphlayer1b_catFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,dummy,*args):
        r=batched_subgraphlayer1b.dummy()
        ctx.args=[x.obj for x in args]
        r.obj=_batched_subgraphlayer1b.cat(ctx.args)
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        offs=0
        dummies=[]
        for x in ctx.args:
            x.add_cat_back(ctx.r,offs)
            offs=offs+x.dim(0)
            dummies.append(batched_subgraphlayer1b.dummy())
        return None, *dummies #it was *dummies


class Batched_subgraphlayer1b_outerFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y):
        r=batched_subgraphlayer1b.dummy()
        r.obj=x.obj.outer(y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        ctx.x.outer_back0(ctx.r,ctx.y)
        ctx.y.outer_back0(ctx.r,ctxxy)
        return batched_subgraphlayer1b.dummy(), batched_subgraphlayer1b.dummy()


class BatchedSubgraphlayer1b_LinmapsFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        r=x.dummy()
        r.obj=_batched_subgraphlayer1b.linmaps(x.obj) 
        ctx.x=x.obj
        ctx.r=r.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        ctx.x.add_linmaps_back(ctx.r)
        return batched_subgraphlayer1b.dummy()


class BatchedSubgraphlayer1b_GatherFromPtensorsbFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,G,S,min_overlaps):
        r=batched_subgraphlayer1b.dummy()
        r.obj=_batched_subgraphlayer1b(x.obj,G.obj,S.obj,min_overlaps)
        ctx.x=x.obj
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_gather_back_alt(ctx.r)
        return batched_subgraphlayer1b.dummy(), None, None, None


class BatchedSubgraphlayer1b_GatherFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,S,min_overlaps):
        r=x.dummy()
        r.obj=_batched_subgraphlayer1b(x.obj,S.obj,min_overlaps)
        ctx.x=x.obj
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_gather_back_alt(ctx.r)
        return batched_subgraphlayer1b.dummy(), None, None  


# this could be shared with subgraphlayer1b
class BatchedSubgraphlayer1b_autobahnFn(torch.autograd.Function):
     @staticmethod
     def forward(ctx,x,w,b):
         r=batched_subgraphlayer1b.dummy()
         r.obj=x.obj.autobahn(w,b)
         ctx.x=x.obj
         ctx.r=r.obj
         ctx.w=w
         ctx.b=b
         return r
     @staticmethod
     def backward(ctx,g):
         wg=torch.zeros_like(ctx.w)
         bg=torch.zeros_like(ctx.b)
         ctx.x.add_autobahn_back0(ctx.r,ctx.w)
         ctx.x.autobahn_back1(wg,bg,ctx.r)
         return batched_subgraphlayer1b.dummy(),wg,bg


# class Batched_subgraphlayer1b_fromMxFn(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx,G,x):
#         R=batched_subgraphlayer1b(1)
#         R.obj=_batched_subgraphlayer1b(G.obj,x)
#         ctx.r=R.obj
#         return R

#     @staticmethod
#     def backward(ctx,g):
#         return None, ctx.r.get_grad().torch()


