#from ptens.ptensors0 import ptensors0
#from ptens.ptensors1 import ptensors1
#from ptens.ptensors2 import ptensors2
from typing import Callable, Optional, Tuple, Union
import torch

import ptens_base
from ptens_base import ptensors0 as _ptensors0
from ptens_base import ptensors1 as _ptensors1
from ptens_base import ptensors2 as _ptensors2

import ptens.ptensorX

def _ptensorsX(order: int) -> Union[_ptensors0,_ptensors1,_ptensors2]:
    return [_ptensors0,_ptensors1,_ptensors2][order]

class ptensorsX(torch.Tensor):
    def __init__(self, order: int, obj: Union[_ptensors0,_ptensors1,_ptensors2]):
        super().__init__(1)
        self.obj = obj
        self.order = order
    @staticmethod
    def from_matrix(T: torch.Tensor, atoms = None, order: int = 0):
        return PtensorsX_fromMxFn.apply(T, atoms, order)
    @staticmethod
    def dummy(order: int):
        return ptensorsX(order,_ptensorsX(order).dummy())

    @staticmethod
    def raw(order: int, _atoms, _nc: int, _dev=0):
        return ptensorsX(order,_ptensorsX(order).raw(_atoms,_nc,_dev))

    @staticmethod
    def zeros(order: int, _atoms, _nc: int, _device='cpu'):
        return ptensorsX(order,_ptensorsX(order).zero(_atoms,_nc,ptens.device_id(_device)))

    @staticmethod
    def randn(order: int, _atoms, _nc: int, _sigma=1.0, _device='cpu'):
        return ptensorsX(order,_ptensorsX(order).zero(_atoms,_nc,ptens.device_id(_device)))


    @staticmethod
    def sequential(order: int, _atoms, _nc, _device='cpu'):
        return ptensorsX(order,_ptensorsX(order).sequential(_atoms,_nc,ptens.device_id(_device)))

    @classmethod
    def randn_like(self,sigma=1.0):
        return ptensorsX.randn(self.order,self.get_atoms(),self.get_nc(),sigma,self.get_dev())


    # ----- Access -------------------------------------------------------------------------------------------


    def _get_grad(self):
        return self.obj.get_grad()
    
    def _view_of_grad(self):
        return self.obj.view_of_grad()
    
    def get_grad(self):
        return ptensorsX(self.order,self.obj.get_grad())
    
    def view_of_grad(self):
        return ptensorsX(self.order,self.obj.view_of_grad())


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
        return PtensorsX_getFn.apply(self,i)
    
    def torch(self):
        return PtensorsX_toMxFn.apply(self)

    def to(self, _device='cpu'):
        self.obj.to_device(ptens.device_id(_device))
        

    # ---- Operations ----------------------------------------------------------------------------------------


    def __add__(self,y):
        return PtensorsX_addFn.apply(self,y)

    def __mul__(self,y):
        return PtensorsX_mprodFn.apply(self,y)

    def linear(self,y,b):
        return PtensorsX_linearFn.apply(self,y,b)

    def concat(self,y):
        return PtensorsX_concatFn.apply(self,y)

    def relu(self,alpha=0.5):
        return PtensorsX_ReLUFn.apply(self,alpha)
        
    def inp(self,y):
        return PtensorsX_inpFn.apply(self,y)
    
    def diff2(self,y):
        return PtensorsX_diff2Fn.apply(self,y)
    
    def linmaps(self, target_order: int):
        return PtensorsX_LinmapsYFn.apply(self,target_order)

    def transfer(self,_atoms, target_order: int):
        return PtensorsX_TransferYFn.apply(self,_atoms, target_order)

    def unite(self,G, target_order: int):
        return PtensorsX_UniteYFn.apply(self,G,target_order)
    
    def gather(self,G):
        return PtensorsX_GatherFn.apply(self,G)

    def outer(self,y):
        return PtensorsX_OuterFn.apply(self,y)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ------------------------------------------------------------------------------------------------------------

# ----------------- Unary Operators ----------------------------------------- #

class PtensorsX_fromMxFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x : torch.Tensor, order: int, atoms = None) -> ptensorsX:
        R = ptensorsX(order,_ptensorsX(order)(x))
        ctx.r = R.obj
        return R

    @staticmethod
    def backward(ctx,g: ptensorsX) -> Tuple[ptensorsX,None,Optional[None]]:
        return (ctx.r.get_grad().torch(), None) if g.order == 0 else (ctx.r.get_grad().torch(), None, None)

class PtensorsX_toMxFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x: ptensorsX):
        ctx.x=x.obj
        ctx.order = x.order
        return x.obj.torch()
 
    @staticmethod
    def backward(ctx,g):
       ctx.x.add_to_grad(_ptensorsX(ctx.order)(g,ctx.x.get_atoms()))
       return ptensorsX.dummy(ctx.order)

class PtensorsX_LinmapsYFn(torch.autograd.Function):
# TODO: figure out those constants.
    @staticmethod
    def forward(ctx,x, to_order: int):
        mult_factor = [
            [ 1, 1,  2],
            [ 1, 2,  5],
            [ 2, 5, 15],
        ][x.order,to_order]
        R = ptensorsX.zeros(to_order,x.obj.view_of_atoms(),x.obj.get_nc()*mult_factor,x.obj.get_dev())
        [
            [ptens_base.add_linmaps0to0,ptens_base.add_linmaps0to1,ptens_base.add_linmaps0to2],
            [ptens_base.add_linmaps1to0,ptens_base.add_linmaps1to1,ptens_base.add_linmaps2to2],
            [ptens_base.add_linmaps2to0,ptens_base.add_linmaps2to1,ptens_base.add_linmaps2to2],
        ][x.order,to_order](R.obj,x.obj)
        ctx.x, ctx.r, ctx.order = x.obj, R.obj, x.order
        return R
        
    @staticmethod
    def backward(ctx,g):
        [
            [ptens_base.add_linmaps0to0_back,ptens_base.add_linmaps0to1_back,ptens_base.add_linmaps0to2_back],
            [ptens_base.add_linmaps1to0_back,ptens_base.add_linmaps1to1_back,ptens_base.add_linmaps2to2_back],
            [ptens_base.add_linmaps2to0_back,ptens_base.add_linmaps2to1_back,ptens_base.add_linmaps2to2_back],
        ][ctx.order,g.order](ctx.x.gradp(),ctx.r.gradp())
        return ptensorsX.dummy(ctx.order), None

class PtensorsX_ReLUFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,alpha):
        R = ptensorsX.zeros(x.order,x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
        R.obj.add_ReLU(x.obj,alpha)
        ctx.x, ctx.alpha, ctx.r = x.obj, alpha, R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_ReLU_back(ctx.r.gradp(),ctx.alpha)
        return ptensorsX.dummy(ctx.order), None

# ----------------------- Binary Operators ---------------------------------- #
class PtensorsX_addFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R = ptensorsX(x.order,_ptensorsX(x)(x.obj))
        R.obj.add(y.obj)
        ctx.x, ctx.y, ctx.r, ctx.order = x.obj, y.obj, R.obj, x.order
        return R

    @staticmethod
    def backward(ctx,g):
        grad = ctx.r.get_gradp()
        ctx.x.add_to_grad(grad)
        ctx.y.add_to_grad(grad)
        return ptensorsX.dummy(ctx.order), ptensorsX.dummy(ctx.order)

class PtensorsX_diff2Fn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        ctx.x, ctx.y = x.obj, y.obj
        return torch.tensor(x.obj.diff2(y.obj))

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_to_grad(ctx.x,g.item()*2.0)
        ctx.x.add_to_grad(ctx.y,-g.item()*2.0)
        ctx.y.add_to_grad(ctx.y,g.item()*2.0)
        ctx.y.add_to_grad(ctx.x,-g.item()*2.0)
        return ptensorsX.dummy(ctx.order), ptensorsX.dummy(ctx.order)

class PtensorsX_concatFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R = ptensorsX(x.order,_ptensorsX(x.order)(x.obj,y.obj))
        ctx.x, ctx.y, ctx.r, ctx.order = x.obj, y.obj, R.obj, x.order
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_concat_back(ctx.r,0)
        ctx.y.add_concat_back(ctx.r,ctx.x.get_nc())
        return ptensorsX.dummy(ctx.order), ptensorsX.dummy(ctx.order)

class PtensorsX_mprodFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R = ptensorsX.zeros(x.order,x.obj.view_of_atoms(),y.size(1),x.obj.get_dev())
        R.obj.add_mprod(x.obj,y)
        ctx.x, ctx.y, ctx.r, ctx.order = x.obj, y.obj, R.obj, x.order
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_mprod_back0(ctx.r.gradp(),ctx.y)
        return ptensorsX.dummy(ctx.order), ctx.x.mprod_back1(ctx.r.gradp())

class PtensorsX_OuterFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y):
        # TODO: check if the order is correctly calculated.
        R = ptensorsX((x.order + 1) * (y.order + 1),ptens_base.outer(x.obj,y.obj))
        ctx.x, ctx.y, ctx.r, ctx.order = x.obj, y.obj, R.obj, x.order
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_outer_back0(ctx.x.gradp(),ctx.r.gradp(),ctx.y)
        ptens_base.add_outer_back1(ctx.y.gradp(),ctx.r.gradp(),ctx.x)
        return ptensorsX.dummy(ctx.order), ptensorsX.dummy(ctx.order)


class PtensorsX_inpFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        ctx.x, ctx.y, ctx.order =x.obj, y.obj, x.order 
        return torch.tensor(x.obj.inp(y.obj))

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_to_grad(ctx.y,g.item())
        ctx.y.add_to_grad(ctx.x,g.item())
        return ptensorsX.dummy(ctx.order), ptensorsX.dummy(ctx.order)


# ----------------------- Ternary Operators --------------------------------- #


class PtensorsX_linearFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y,b):
        R = ptensorsX.zeros(x.order,x.obj.view_of_atoms(),y.size(1),x.obj.get_dev())
        R.obj.add_linear(x.obj,y,b)
        ctx.x, ctx.y, ctx.r, ctx.order = x.obj, y.obj, R.obj, x.order
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_linear_back0(ctx.r.gradp(),ctx.y)
        return ptensorsX.dummy(ctx.order), ctx.x.linear_back1(ctx.r.gradp()), ctx.x.linear_back2(ctx.r.gradp())


# ------------------------ Message Passing Related -------------------------- #


class PtensorsX_TransferYFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,atoms,G, target_order: int):
        # hard coded because I'm tired and don't feel like doing math...
        mult_factor = [
            [ 1, 1,  2],
            [ 1, 2,  5],
            [ 2, 5, 15],
        ][x.order,target_order]
        R = ptensorsX.zeros(target_order,atoms,x.obj.get_nc()*mult_factor,x.obj.get_dev())
        ptens_base.add_msg(R.obj,x.obj,G.obj)
        ctx.x, ctx.r, ctx.G, ctx.order = x.obj, R.obj, G.obj, x.order
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_msg_back(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        return ptensorsX.dummy(ctx.order), None, None, None

class PtensorsX_UniteYFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,G, target_order: int):
        R = ptensorsX(target_order,[ptens_base.unite1,ptens_base.unite2][target_order](x.obj,G.obj))
        ctx.x, ctx.r, ctx.G, ctx.order = x.obj, R.obj, G.obj, x.order
        return R
        
    @staticmethod
    def backward(ctx,g):
        [
            [ptens_base.unite0to1_back, ptens_base.unite0to2_back],
            [ptens_base.unite1to1_back, ptens_base.unite1to2_back],
            [ptens_base.unite2to1_back, ptens_base.unite2to2_back],
        ][ctx.order,g.order](ctx.x.gradp,ctx.r.gradp,ctx.G)
        return ptensorsX.dummy(ctx.order), None, None

class PtensorsX_GatherFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,G):
        R = ptensorsX(x.order,ptens_base.gather(x.obj,G.obj))
        ctx.x, ctx.r, ctx.G, ctx.order = x.obj, R.obj, G.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.gather_back(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        return ptensorsX.dummy(ctx.order), None

# ----------------------------- Access -------------------------------------- #
class PtensorsX_getFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,i):
        R = ptensorsX(x.order,x.obj[i].torch())
        R.atoms=x.atoms_of(i)
        ctx.x, ctx.i, ctx.order = x.obj, i, x.order
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_to_grad(ctx.i,g)
        return ptensorsX.dummy(ctx.order), None
