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

from typing import Literal

import torch

import ptens_base 
from ptens_base import subgraph_layer1 as _subgraph_layer1
from ptens.utility import device_id as device_id

import ptens.ptensors0 
import ptens.ptensors1 
import ptens.ptensors2 

from ptens.subgraphlayer import *


class subgraphlayer1(subgraphlayer):

    @classmethod
    def from_matrix(self,G,S,atoms,M):
        return SubgraphLayer1_fromMxFn.apply(G,S,atoms,M)

    @classmethod
    def like(self,x,M):
        return SubgraphLayer1_likeFn.apply(x,M)

    @classmethod
    def dummy(self):
        R=subgraphlayer1(1)
        #R.obj=_subgraphlayer1.dummy()
        return R

    @classmethod
    def raw(self,G,S,_atoms,_nc,device=0):
        R=subgraphlayer1(1)
        R.obj=_subgraph_layer1.raw(G.obj,S.obj,_atoms,_nc,ptens.device_id(device))
        return R

    @classmethod
    def zeros(self,G,S,_atoms,_nc,device=0):
        R=subgraphlayer1(1)
        R.obj=_subgraph_layer1.zero(G.obj,S.obj,_atoms,_nc,ptens.device_id(device))
        return R

    @classmethod
    def randn(self,G,S,_atoms,_nc,device=0):
        R=subgraphlayer1(1)
        R.obj=_subgraph_layer1.gaussian(G.obj,S.obj,_atoms,_nc,ptens.device_id(device))
        return R

    @classmethod
    def sequential(self,G,S,_atoms,_nc,device=0):
        R=subgraphlayer1(1)
        R.obj=_subgraph_layer1.sequential(G.obj,S.obj,_atoms,_nc,ptens.device_id(device))
        return R

    @classmethod
    def gather(self,x,S):
        return SubgraphLayer1_GatherFn.apply(x,S)

    @classmethod
    def gather_from_ptensors(self,x,G,S):
        return SubgraphLayer1_GatherFromPtensorsFn.apply(x,G,S)


    # ----- Access -------------------------------------------------------------------------------------------


    def _get_grad(self):
        return self.obj.get_grad()
    
    def _view_of_grad(self):
        return self.obj.view_of_grad()
    
    def get_grad(self):
        R=subgraphlayer1(1)
        R.obj=self.obj.get_grad()
        return R
    
    def view_of_grad(self):
        R=subgraphlayer1(1)
        R.obj=self.obj.view_of_grad()
        return R


    def get_dev(self):
        return self.obj.get_dev()

    def get_nc(self):
        return self.obj.get_nc()

    def get_atoms(self):
        return self.obj.get_atoms()
    
    def atoms_of(self, i):
        return self.obj.atoms_of(i)

    def __getitem__(self,i):
        return Subgraph_layer1_getFn.apply(self,i)
    
    def push_back(self, x):
        return self.obj.push_back(x)

    def randn_like(self,sigma=1.0):
        return subgraphlayer1.randn(self.get_atoms(),self.get_nc(),sigma,self.get_dev())

    def torch(self):
        return SubgraphLayer1_toMxFn.apply(self)

    def ptensors1(self):
        return SubgraphLayer1_toPtensors1Fn.apply(self)
    
    def to(self, device='cpu'):
        return SubgraphLayer_toDeviceFn.apply(self,device)


    # ---- Operations ----------------------------------------------------------------------------------------

    
    def __add__(self,y):
        return SubgraphLayer_plusFn.apply(self,y)

    def __mul__(self,y):
        return SubgraphLayer_mprodFn.apply(self,y)

    def linear(self,y,b):
        return SubgraphLayer_linearFn.apply(self,y,b)

    def concat(self,y):
        return SubgraphLayer_concatFn.apply(self,y)
    
    def relu(self,alpha=0.1):
        return SubgraphLayer_ReLUFn.apply(self,alpha)
        
    def inp(self,y):
        return SubgraphLayer_inpFn.apply(self,y)
    
    def diff2(self,y):
        return ptens.SubgraphLayer1_diff2Fn.apply(self,y)

    def outer(self,y):
        if isinstance(y,ptens.ptensors0):
            return Subgraph_layer1_Outer0Fn.apply(self,y)
        if isinstance(y,ptens.subgraphlayer1):
            return Subgraph_layer1_Outer1Fn.apply(self,y)

    def scale(self,y):
        return Subgraph_layer1_scaleFn.apply(self,y)

    def mult_channels(self,y):
        return Subgraph_layer1_mult_channelsFn.apply(self,y)


    # ---- Message passing -----------------------------------------------------------------------------------


#     def linmaps0(self,normalized=False):
#         return Subgraph_layer1_Linmaps0Fn.apply(self,normalized);

#     def linmaps1(self,normalized=False):
#         return Subgraph_layer1_Linmaps1Fn.apply(self,normalized);

#     def linmaps2(self,normalized=False):
#         return Subgraph_layer1_Linmaps2Fn.apply(self,normalized);


    def autobahn(self,w,b):
        return Subgraph_layer1_autobahnFn.apply(self,w,b)


    def transfer0(self,_atoms,G,normalized=False):
        return Subgraph_layer1_Transfer0Fn.apply(self,_atoms,G,normalized)

    def transfer1(self,_atoms,G,normalized=False):
        return Subgraph_layer1_Transfer1Fn.apply(self,_atoms,G,normalized)

    def transfer2(self,_atoms,G,normalized=False):
        return Subgraph_layer1_Transfer2Fn.apply(self,_atoms,G,normalized)
    
    
    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ------------------------------------------------------------------------------------------------------------


class SubgraphLayer1_fromMxFn(torch.autograd.Function): #TODO 
    @staticmethod
    def forward(ctx,G,x):
        R=subgraphlayer1(1)
        R.obj=_subgraph_layer1(G.obj,x)
        return R
    @staticmethod
    def backward(ctx,g):
        return None, ctx.r.get_grad().torch()


class SubgraphLayer1_toMxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        ctx.x=x.obj
        return x.obj.torch()
    @staticmethod
    def backward(ctx,g):
       R=subgraphlayer1(1)
       ctx.x.torch_back(g)
       return R
    

class SubgraphLayer1_likeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,M):
        r=subgraphlayer1(1)
        r.obj=_subgraph_layer1.like(x.obj,M)
        ctx.r=r.obj
        return r
    @staticmethod
    def backward(ctx,g):
        return None, ctx.r.get_grad().torch()



# class SubgraphLayer1_toPtensors1Fn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x):
#         r=ptensors1(1)
#         r.obj=x.obj.ptensors1()
#         ctx.x=x.obj
#         ctx.r=r.obj
#         return r
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.toPtensors1_back(ctx.r)
#         return subgraphlayer1.dummy()


# class Subgraph_layer1_toFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,_dev):
#         dev=ptens.device_id(_dev)
#         R=subgraphlayer1(1)
#         R.obj=_subgraph_layer1(x.obj,dev)
#         ctx.x=x.obj
#         ctx.r=R.obj
#         ctx.dev=dev
#         return R
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.move_to_device_back(ctx.r.get_gradp(),ctx.dev)
#         return subgraphlayer1.dummy(), None
        
    
# class SubgraphLayer1_addFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,y):
#         R=subgraphlayer1(1)
#         R.obj=_subgraph_layer1(x.obj)
#         R.obj.add(y.obj)
#         ctx.x=x.obj
#         ctx.y=y.obj
#         ctx.r=R.obj
#         return R
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.add_to_grad(ctx.r.get_gradp())
#         ctx.y.add_to_grad(ctx.r.get_gradp())
#         return subgraphlayer1(1),subgraphlayer1(1)


# class SubgraphLayer1_ReLUFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,alpha):
#         r=subgraphlayer1(1)
#         r.obj=x.obj.ReLU(alpha)
#         ctx.x=x.obj
#         ctx.alpha=alpha
#         ctx.r=r.obj
#         return r
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.add_ReLU_back(ctx.r,ctx.alpha)
#         return subgraphlayer1.dummy(), None

# class SubgraphLayer1_inpFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,y):
#         ctx.x=x.obj
#         ctx.y=y.obj
#         return torch.tensor(x.obj.inp(y.obj))
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.add_to_grad(ctx.y,g.item())
#         ctx.y.add_to_grad(ctx.x,g.item())
#         return subgraphlayer1.dummy(), subgraph_layer1.dummy()

# class Ptensors0_diff2Fn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,y):
#         ctx.x=x.obj
#         ctx.y=y.obj
#         return torch.tensor(x.obj.diff2(y.obj))
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.add_to_grad(ctx.x,g.item()*2.0)
#         ctx.x.add_to_grad(ctx.y,-g.item()*2.0)
#         ctx.y.add_to_grad(ctx.y,g.item()*2.0)
#         ctx.y.add_to_grad(ctx.x,-g.item()*2.0)
#         return subgraphlayer1.dummy(), subgraphlayer1.dummy()


# class Subgraph_layer1_concatFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,y):
#         r=subgraphlayer1(1)
#         r.obj=_subgraph_layer1.concat(x.obj,y.obj)
#         ctx.x=x.obj
#         ctx.y=y.obj
#         ctx.r=r.obj
#         return r
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.add_concat_back(ctx.r,0)
#         ctx.y.add_concat_back(ctx.r,ctx.x.get_nc())
#         return subgraphlayer1(1),subgraphlayer1(1)


# class Subgraph_layer1_mprodFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,y):
#         R=ptens.subgraphlayer1.zeros(x.obj.view_of_atoms(),y.size(1),x.obj.get_dev())
#         R.obj.add_mprod(x.obj,y)
#         ctx.x=x.obj
#         ctx.y=y
#         ctx.r=R.obj
#         return R
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.add_mprod_back0(ctx.r.gradp(),ctx.y)
#         return subgraphlayer1(1), ctx.x.mprod_back1(ctx.r.gradp())


# class Subgraph_layer1_linearFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,y,b):
#         R=ptens.subgraphlayer1.zeros(x.obj.view_of_atoms(),y.size(1),x.obj.get_dev())
#         R.obj.add_linear(x.obj,y,b)
#         ctx.x=x.obj
#         ctx.y=y
#         ctx.r=R.obj
#         return R
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.add_linear_back0(ctx.r.gradp(),ctx.y)
#         return subgraphlayer1.dummy(), ctx.x.linear_back1(ctx.r.gradp()), ctx.x.linear_back2(ctx.r.gradp())


class Subgraph_layer1_autobahnFn(torch.autograd.Function):
     @staticmethod
     def forward(ctx,x,w,b):
         r=ptens.subgraphlayer1(1)
         r.obj=x.obj.autobahn(w,b)
         ctx.x=x.obj
         ctx.w=w
         ctx.b=b
         return r
     @staticmethod
     def backward(ctx,g):
         wg=torch.zeros_like(ctx.w)
         bg=torch.zeros_like(ctx.b)
         ctx.x.add_autobahn_back0(ctx.r,ctx.w)
         ctx.x.add_autobahn_back1(wg,bg,ctx.r)
         return subgraphlayer1.dummy(),wg,bg


class Subgraph_layer1_scaleFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptens.ptensors1.zeros(x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
        R.obj.add_scale(x.obj,y)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_scale_back0(ctx.r.gradp(),ctx.y)
        return ptensors0.dummy(), ctx.x.scale_back1(ctx.r.gradp())


class Subgraph_layer1_mult_channelsFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptens.subgraphlayer1.zeros(x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
        R.obj.add_scale_channels(x.obj,y)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_scale_channels_back0(ctx.r.gradp(),ctx.y)
        return subgraphlayer1.dummy(), None


class Subgraph_layer1_Linmaps0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,normalized=False):
        R=ptens.ptensors0.zeros(x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
        if(normalized):
            ptens_base.add_linmaps1to0_n(R.obj,x.obj)
        else:
            ptens_base.add_linmaps1to0(R.obj,x.obj)
        ctx.normalized=normalized
        ctx.x=x.obj
        ctx.r=R.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        if(ctx.normalized):
            ptens_base.add_linmaps1to0_back_n(ctx.x.gradp(),ctx.r.gradp())
        else:
            ptens_base.add_linmaps1to0_back(ctx.x.gradp(),ctx.r.gradp())
        return subgraphlayer1(1), None


class Subgraph_layer1_Linmaps1Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,normalized=False):
        R=ptens.subgraphlayer1.zeros(x.obj.view_of_atoms(),x.obj.get_nc()*2,x.obj.get_dev())
        ptens_base.add_linmaps1to1(R.obj,x.obj)
        ctx.normalized=normalized
        ctx.x=x.obj
        ctx.r=R.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        if(ctx.normalized):
            ptens_base.add_linmaps1to1_back_n(ctx.x.gradp(),ctx.r.gradp())
        else:
            ptens_base.add_linmaps1to1_back(ctx.x.gradp(),ctx.r.gradp())
        return subgraphlayer1(1), None


class Subgraph_layer1_Linmaps2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,normalized=False):
        R=ptens.ptensors2.zeros(x.obj.view_of_atoms(),x.obj.get_nc()*5,x.obj.get_dev())
        if(normalized):
            ptens_base.add_linmaps1to2_n(R.obj,x.obj)
        else:
            ptens_base.add_linmaps1to2(R.obj,x.obj)
        ctx.normalized=normalized
        ctx.x=x.obj
        ctx.r=R.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        if(ctx.normalized):
            ptens_base.add_linmaps1to2_back_n(ctx.x.gradp(),ctx.r.gradp())
        else:
            ptens_base.add_linmaps1to2_back(ctx.x.gradp(),ctx.r.gradp())
        return subgraphlayer1(1), None




class SubgraphLayer1_GatherFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,S):
        ctx.x=x
        r=ptens.subgraphlayer1(1)
        r.obj=_subgraph_layer1(x.obj,S.obj)
        ctx.x=x.obj
        ctx.r=r.obj
        return r
    @staticmethod
    def backward(ctx,g):
        ctx.r.gather_back(ctx.x)
        return subgraphlayer1.dummy(), None, None


class SubgraphLayer1_GatherFromPtensorsFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,G,S):
        ctx.x=x
        r=ptens.subgraphlayer1(1)
        r.obj=_subgraph_layer1(x.obj,G.obj,S.obj)
        ctx.x=x.obj
        ctx.r=r.obj
        return r
    @staticmethod
    def backward(ctx,g):
        ctx.r.gather_back(ctx.x)
        return subgraphlayer1.dummy(), None, None, None 





# class Subgraph_layer1_Outer0Fn(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx,x,y):
#         r=ptens.subgraphlayer1(1)
#         r.obj=ptens_base.outer(x.obj,y.obj)
#         ctx.x=x.obj
#         ctx.y=y.obj
#         ctx.r=r.obj
#         return r
        
#     @staticmethod
#     def backward(ctx,g):
#         ptens_base.add_outer_back0(ctx.x.gradp(),ctx.r.gradp(),ctx.y)
#         ptens_base.add_outer_back1(ctx.y.gradp(),ctx.r.gradp(),ctx.x)
#         return subgraphlayer1.dummy(), ptens.ptensors0.dummy()


# class Subgraph_layer1_Outer1Fn(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx,x,y):
#         r=ptens.ptensors2(1)
#         r.obj=ptens_base.outer(x.obj,y.obj)
#         ctx.x=x.obj
#         ctx.y=y.obj
#         ctx.r=r.obj
#         return r
        
#     @staticmethod
#     def backward(ctx,g):
#         ptens_base.add_outer_back0(ctx.x.gradp(),ctx.r.gradp(),ctx.y)
#         ptens_base.add_outer_back1(ctx.y.gradp(),ctx.r.gradp(),ctx.x)
#         return subgraphlayer1.dummy(), subgraphlayer1.dummy()









# class Subgraph_layer1_getFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,i):
#         R=ptens.ptensor1(x.obj[i].torch())
#         R.atoms=x.atoms_of(i)
#         ctx.x=x.obj
#         ctx.i=i
#         return R
#     @staticmethod
#     def backward(ctx,g):
#         R=subgraphlayer1(1)
#         ctx.x.add_to_grad(ctx.i,g)
#         return R,None
