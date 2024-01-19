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
from ptens_base import ptensors0 as _ptensors0
from ptens.utility import device_id as device_id

import ptens.ptensor0
import ptens.ptensors1
import ptens.ptensors2 


class ptensors0(torch.Tensor):

    @classmethod
    def from_matrix(self,T,atoms=None):
        return Ptensors0_fromMxFn.apply(T,atoms)
            
    @classmethod
    def dummy(self):
        R=ptensors0(1)
        R.obj=_ptensors0.dummy()
        return R

    @classmethod
    def raw(self, _atoms, _nc, device='cpu'):
        R=ptensors0(1)
        R.obj=_ptensors0.raw(_atoms,_nc,ptens.device_id(device))
        return R

    @classmethod
    def zeros(self, _atoms, _nc, device='cpu'):
        R=ptensors0(1)
        R.obj=_ptensors0.zero(_atoms,_nc,ptens.device_id(device))
        return R

    @classmethod
    def randn(self, _atoms, _nc, _sigma=1.0, device='cpu'):
        R=ptensors0(1)
        R.obj=_ptensors0.gaussian(_atoms,_nc,_sigma,ptens.device_id(device))
        return R

    @classmethod
    def sequential(self, _atoms, _nc, device='cpu'):
        R=ptensors0(1)
        R.obj=_ptensors0.sequential(_atoms,_nc,ptens.device_id(device))
        return R

    @classmethod
    def cat(self,*args):
        return Ptensors0_catFn.apply(self,*args)

    @classmethod
    def sum(self,*args):
        return Ptensors0_sumFn.apply(self,*args)

    def randn_like(self,sigma=1.0):
        return ptensors0.randn(self.get_atoms(),self.get_nc(),sigma,self.get_dev())


    # ----- Access -------------------------------------------------------------------------------------------


    def __len__(self):
        return len(self.obj)
    
    def _get_grad(self):
        return self.obj.get_grad()
    
    def _view_of_grad(self):
        return self.obj.view_of_grad()
    
    def get_grad(self):
        R=ptensors0(1)
        R.obj=self.obj.get_grad()
        return R
    
    def view_of_grad(self):
        R=ptensors0(1)
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

    def push_back(self, x):
        return self.obj.push_back(x)

    def __getitem__(self,i):
        return Ptensors0_getFn.apply(self,i)
    
    def torch(self):
        return Ptensors0_toMxFn.apply(self)

    def to(self, device='cpu'):
        return Ptensors0_toFn.apply(self,device)
        #self.obj.to_device(ptens.device_id(device))
        

    # ---- Operations ----------------------------------------------------------------------------------------


    def __add__(self,y):
        return Ptensors0_addFn.apply(self,y)

    def __mul__(self,y):
        return Ptensors0_mprodFn.apply(self,y)

    def linear(self,y,b):
        return Ptensors0_linearFn.apply(self,y,b)

    def concat(self,y):
        return Ptensors0_concatFn.apply(self,y)

    def relu(self,alpha=0.5):
        return Ptensors0_ReLUFn.apply(self,alpha)
        
    def inp(self,y):
        return Ptensors0_inpFn.apply(self,y)
    
    def diff2(self,y):
        return Ptensors0_diff2Fn.apply(self,y)

    def outer(self,y):
        if isinstance(y,ptens.ptensors0):
            return Ptensors0_Outer0Fn.apply(self,y)
        if isinstance(y,ptens.ptensors1):
            return Ptensors0_Outer1Fn.apply(self,y)
        if isinstance(y,ptens.ptensors2):
            return Ptensors0_Outer2Fn.apply(self,y)

    def scale(self,y):
        return Ptensors0_scaleFn.apply(self,y)

    def mult_channels(self,y):
        return Ptensors0_mult_channelsFn.apply(self,y)

    def avg_pool(self):
        return Ptensors0_averageFn.apply(self)


    # ---- Message passing -----------------------------------------------------------------------------------
    

    def linmaps0(self,normalized=False):
        return Ptensors0_Linmaps0Fn.apply(self);

    def linmaps1(self,normalized=False):
        return Ptensors0_Linmaps1Fn.apply(self);

    def linmaps2(self,normalized=False):
        return Ptensors0_Linmaps2Fn.apply(self);


    def transfer0(self,_atoms,G,normalized=False):
        return Ptensors0_Transfer0Fn.apply(self,_atoms,G)

    def transfer1(self,_atoms,G,normalized=False):
        return Ptensors0_Transfer1Fn.apply(self,_atoms,G)

    def transfer2(self,_atoms,G,normalized=False):
        return Ptensors0_Transfer2Fn.apply(self,_atoms,G)


    def unite1(self,G,normalized=False):
        return Ptensors0_Unite1Fn.apply(self,G)
    
    def unite2(self,G,normalized=False):
        return Ptensors0_Unite2Fn.apply(self,G)
    
    def gather(self,G,normalized=False):
        return Ptensors0_GatherFn.apply(self,G)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ------------------------------------------------------------------------------------------------------------


class Ptensors0_fromMxFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,atoms):
        R=ptensors0(1)
        if atoms:
            R.obj=_ptensors0(x,atoms)
        else:
            R.obj=_ptensors0(x)
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        return ctx.r.get_grad().torch(), None


class Ptensors0_toMxFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        ctx.x=x.obj
        return x.obj.torch()
 
    @staticmethod
    def backward(ctx,g):
       R=ptensors0(1)
       ctx.x.add_to_grad(_ptensors0(g))
       return R
    

class Ptensors0_getFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,i):
        R=ptens.ptensor0(x.obj[i].torch())
        R.atoms=x.atoms_of(i)
        ctx.x=x.obj
        ctx.i=i
        return R

    @staticmethod
    def backward(ctx,g):
        R=ptensors0(1)
        ctx.x.add_to_grad(ctx.i,g)
        return R,None


class Ptensors0_toFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,_dev):
        dev=ptens.device_id(_dev)
        R=ptensors0(1)
        R.obj=_ptensors0(x.obj,dev)
        ctx.x=x.obj
        ctx.r=R.obj
        ctx.dev=dev
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.move_to_device_back(ctx.r.get_gradp(),ctx.dev)
        return ptensors0.dummy(), None
        

class Ptensors0_addFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptensors0(1)
        _ptensors0(x.obj)
        R.obj=_ptensors0(x.obj)
        R.obj.add(y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_to_grad(ctx.r.get_gradp())
        ctx.y.add_to_grad(ctx.r.get_gradp())
        return ptensors0.dummy(),ptensors0.dummy()


class Ptensors0_inpFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        ctx.x=x.obj
        ctx.y=y.obj
        return torch.tensor(x.obj.inp(y.obj))

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_to_grad(ctx.y,g.item())
        ctx.y.add_to_grad(ctx.x,g.item())
        return ptensors0.dummy(), ptensors0.dummy()


class Ptensors0_diff2Fn(torch.autograd.Function):
    
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
        return ptensors0.dummy(), ptensors0.dummy()


class Ptensors0_concatFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        r=ptensors0(1)
        r.obj=_ptensors0.concat(x.obj,y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_concat_back(ctx.r,0)
        ctx.y.add_concat_back(ctx.r,ctx.x.get_nc())
        return ptensors0(1),ptensors0(1)

    
class Ptensors0_catFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,dummy,*args):
        r=ptensors1(1)
        ctx.args=[x.obj for x in args]
        r.obj=_ptensors0.cat(ctx.args)
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        offs=0
        dummies=[]
        for x in ctx.args:
            x.add_cat_back(ctx.r,offs)
            offs=offs+len(x)
            dummies.append(ptensors1(1))
        return None, *dummies


class Ptensors0_sumFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,dummy,*args):
        r=ptensors0(1)
        ctx.args=[x.obj for x in args]
        r.obj=_ptensors0.sum(ctx.args)
        ctx.r=r.obj
        return r
    @staticmethod
    def backward(ctx,g):
        dummies=[]
        for x in ctx.args:
            x.add_to_grad(ctx.r.get_gradp())
            dummies.append(ptensors0(1))
        return None, *dummies


class Ptensors0_averageFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        r=ptensors0(1)
        r.obj=_ptensors0.average(x.obj)
        ctx.x=x.obj
        ctx.r=r.obj
        return r
    @staticmethod
    def backward(ctx,g):
        ctx.x.add_average_back(ctx.r)
        return ptensors0(1)


class Ptensors0_mprodFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptens.ptensors0.zeros(x.obj.view_of_atoms(),y.size(1),x.obj.get_dev())
        R.obj.add_mprod(x.obj,y)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_mprod_back0(ctx.r.gradp(),ctx.y)
        return ptensors0.dummy(), ctx.x.mprod_back1(ctx.r.gradp())


class Ptensors0_mult_channelsFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptens.ptensors0.zeros(x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
        R.obj.add_scale_channels(x.obj,y)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_scale_channels_back0(ctx.r.gradp(),ctx.y)
        return ptensors0.dummy(), None


class Ptensors0_scaleFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptens.ptensors0.zeros(x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
        R.obj.add_scale(x.obj,y)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_scale_back0(ctx.r.gradp(),ctx.y)
        return ptensors0.dummy(), ctx.x.scale_back1(ctx.r.gradp())


class Ptensors0_linearFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y,b):
        R=ptens.ptensors0.zeros(x.obj.view_of_atoms(),y.size(1),x.obj.get_dev())
        R.obj.add_linear(x.obj,y,b)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_linear_back0(ctx.r.gradp(),ctx.y)
        return ptensors0.dummy(), ctx.x.linear_back1(ctx.r.gradp()), ctx.x.linear_back2(ctx.r.gradp())



class Ptensors0_ReLUFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,alpha):
        R=ptens.ptensors0.zeros(x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
        R.obj.add_ReLU(x.obj,alpha)
        ctx.x=x.obj
        ctx.alpha=alpha
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_ReLU_back(ctx.r.gradp(),ctx.alpha)
        return ptensors0.dummy(), None




class Ptensors0_Linmaps0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptens.ptensors0.zeros(x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
        ptens_base.add_linmaps0to0(R.obj,x.obj)
        ctx.x=x.obj
        ctx.r=R.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_linmaps0to0_back(ctx.x.gradp(),ctx.r.gradp())
        return ptensors0.dummy()


class Ptensors0_Linmaps1Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptens.ptensors1.zeros(x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
        ptens_base.add_linmaps0to1(R.obj,x.obj)
        ctx.x=x.obj
        ctx.r=R.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_linmaps0to1_back(ctx.x.gradp(),ctx.r.gradp())
        return ptensors0.dummy()


class Ptensors0_Linmaps2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptens.ptensors2.zeros(x.obj.view_of_atoms(),x.obj.get_nc()*2,x.obj.get_dev())
        ptens_base.add_linmaps0to2(R.obj,x.obj)
        ctx.x=x.obj
        ctx.r=R.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_linmaps0to2_back(ctx.x.gradp(),ctx.r.gradp())
        return ptensors0.dummy()


class Ptensors0_Transfer0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,atoms,G):
        ctx.x=x
        R=ptens.ptensors0.zeros(atoms,x.obj.get_nc(),x.obj.get_dev())
        ptens_base.add_msg(R.obj,x.obj,G.obj)
        ctx.x=x.obj
        ctx.r=R.obj
        ctx.G=G.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_msg_back(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        return ptensors0.dummy(), None, None


class Ptensors0_Transfer1Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,atoms,G):
        ctx.x=x
        R=ptens.ptensors1.zeros(atoms,x.obj.get_nc(),x.obj.get_dev())
        ptens_base.add_msg(R.obj,x.obj,G.obj)
        ctx.x=x.obj
        ctx.r=R.obj
        ctx.G=G.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_msg_back(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        return ptensors0.dummy(), None, None


class Ptensors0_Transfer2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,atoms,G):
        ctx.x=x
        R=ptens.ptensors2.zeros(atoms,x.obj.get_nc()*2,x.obj.get_dev())
        ptens_base.add_msg(R.obj,x.obj,G.obj)
        ctx.x=x.obj
        ctx.r=R.obj
        ctx.G=G.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_msg_back(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        return ptensors0.dummy(), None, None


class Ptensors0_Unite1Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,G):
        r=ptens.ptensors1(1)
        r.obj=ptens_base.unite1(x.obj,G.obj)
        ctx.x=x.obj
        ctx.r=r.obj
        ctx.G=G.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.unite0to1_back(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        return ptensors0.dummy(), None


class Ptensors0_Unite2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,G):
        r=ptens.ptensors2(1)
        r.obj=ptens_base.unite2(x.obj,G.obj)
        ctx.x=x.obj
        ctx.r=r.obj
        ctx.G=G.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.unite0to2_back(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        return ptensors0.dummy(), None


class Ptensors0_GatherFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,G):
        r=ptens.ptensors0(1)
        r.obj=ptens_base.gather(x.obj,G.obj)
        ctx.x=x.obj
        ctx.r=r.obj
        ctx.G=G.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.gather_back(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        return ptensors0.dummy(), None


class Ptensors0_Outer0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y):
        r=ptens.ptensors0(1)
        r.obj=ptens_base.outer(x.obj,y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_outer_back0(ctx.x.gradp(),ctx.r.gradp(),ctx.y)
        ptens_base.add_outer_back1(ctx.y.gradp(),ctx.r.gradp(),ctx.x)
        return ptensors0.dummy(), ptensors0.dummy()


class Ptensors0_Outer1Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y):
        r=ptens.ptensors1(1)
        r.obj=ptens_base.outer(x.obj,y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_outer_back0(ctx.x.gradp(),ctx.r.gradp(),ctx.y)
        ptens_base.add_outer_back1(ctx.y.gradp(),ctx.r.gradp(),ctx.x)
        return ptensors0.dummy(), ptens.ptensors1.dummy()


class Ptensors0_Outer2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y):
        r=ptens.ptensors2(1)
        r.obj=ptens_base.outer(x.obj,y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_outer_back0(ctx.x.gradp(),ctx.r.gradp(),ctx.y)
        ptens_base.add_outer_back1(ctx.y.gradp(),ctx.r.gradp(),ctx.x)
        return ptensors0.dummy(), ptens.ptensors2.dummy()






