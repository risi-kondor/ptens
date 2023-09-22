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
#from ptens_base import subgraph_layer1 as _subgraph_layer1
from ptens.utility import device_id as device_id

#import ptens.ptensors0 
#import ptens.ptensors1 
#import ptens.ptensors2 


class subgraphlayer(torch.Tensor):

    @classmethod
    def dummy(self):
        R=subgraphlayer(1)
        return R


# ------------------------------------------------------------------------------------------------------------


class SubgraphLayer_toDeviceFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,_dev):
        dev=ptens.device_id(_dev)
        r=x.dummy()
        r.obj=x.obj.to_device(dev)
        ctx.x=x.obj
        ctx.r=r.obj
        ctx.dev=dev
        return r
    @staticmethod
    def backward(ctx,g):
        ctx.x.to_device_back(ctx.r,ctx.dev)
        return subgraphlayer.dummy(), None


class SubgraphLayer_concatFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y):
        r=x.dummy()
        r.obj=x.obj.concat(y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r
    @staticmethod
    def backward(ctx,g):
        ctx.x.add_concat_back(ctx.r,0)
        ctx.y.add_concat_back(ctx.r,ctx.x.get_nc())
        return subgraphlayer0(1),subgraphlayer0(1)


class SubgraphLayer_plusFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y):
        r=x.dummy()
        r.obj=x.obj.plus(y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r
    @staticmethod
    def backward(ctx,g):
        ctx.x.add_to_grad(ctx.r.gradp())
        ctx.y.add_to_grad(ctx.r.gradp())
        return subgraphlayer.dummy(),subgraphlayer.dummy()


class SubgraphLayer_mprodFn(torch.autograd.Function):
     @staticmethod
     def forward(ctx,x,y):
         r=x.dummy()
         r.obj=x.obj.mprod(y.obj)
         ctx.x=x.obj
         ctx.y=y
         ctx.r=r.obj
         return r
     @staticmethod
     def backward(ctx,g):
         ctx.x.add_mprod_back0(ctx.r.gradp(),ctx.y)
         return subgraphlayer.dummy(), ctx.x.mprod_back1(ctx.r.gradp())


class SubgraphLayer_linearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y,b):
        r=x.dummy()
        r.obj=x.obj.linear(y,b)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=r.obj
        return r
    @staticmethod
    def backward(ctx,g):
        ctx.x.add_linear_back0(ctx.r,ctx.y)
        return subgraphlayer.dummy(), ctx.x.linear_back1(ctx.r), ctx.x.linear_back2(ctx.r)


class SubgraphLayer_inpFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y):
        ctx.x=x.obj
        ctx.y=y.obj
        return torch.tensor(x.obj.inp(y.obj))
    @staticmethod
    def backward(ctx,g):
        ctx.x.add_to_grad(ctx.y,g.item())
        ctx.y.add_to_grad(ctx.x,g.item())
        return subgraphlayer.dummy(), subgraphlayer.dummy()


class SubgraphLayer_diff2Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y):
        ctx.x=x.obj
        ctx.y=y.obj
        return torch.tensor(x.obj.diff2(y.obj))
    @staticmethod
    def backward(ctx,g):
        ctx.x.add_to_grad(ctx.x,g.item()*2.0)
        ctx.x.add_to_grad(ctx.y,-g.item()*2.0)
        ctx.y.add_to_grad(ctx.y,g.item()*2.0)
        ctx.y.add_to_grad(ctx.x,-g.item()*2.0)
        return subgraphlayer.dummy(), subgraphlayer.dummy()


class SubgraphLayer_ReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,alpha):
        r=x.dummy()
        r.obj=x.obj.ReLU(alpha)
        ctx.x=x.obj
        ctx.alpha=alpha
        ctx.r=r.obj
        return r
    @staticmethod
    def backward(ctx,g):
        ctx.x.add_ReLU_back(ctx.r,ctx.alpha)
        return subgraphlayer.dummy(), None
