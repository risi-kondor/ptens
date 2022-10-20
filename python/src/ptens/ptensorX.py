import torch

import ptens_base 
from ptens_base import ptensor0 as _ptensor0
from ptens_base import ptensor1 as _ptensor1
from ptens_base import ptensor2 as _ptensor2

def _ptensorX(order: int):
    return [_ptensor0,_ptensor1,_ptensor2][order]

class ptensorX(torch.Tensor):
    def __init__(self, order: int, value: torch.Tensor, atoms):
        super().__init__(value)
        self.atoms = atoms
        self.order = order
    @staticmethod
    def zeros(order: int, _atoms, _nc):
        return ptensorX(order, torch.zeros(len(_atoms),len(_atoms),_nc),_atoms)
    
    @staticmethod
    def randn(order: int, _atoms, _nc):
        return ptensorX(order, torch.randn(len(_atoms),len(_atoms),_nc),_atoms)

    @staticmethod
    def sequential(order: int, _atoms, _nc):
        return ptensorX(order, torch.randn(len(_atoms),len(_atoms),_nc),_atoms)


    # ---- Access --------------------------------------------------------------------------------------------


    def get_nc(self):
        return self.size(2)


    # ---- Operations ----------------------------------------------------------------------------------------

    
    def linmaps(self, target_order: int):
        return PtensorX_LinmapsYFn.apply(self, target_order)

    def transfer(self,_atoms, target_order: int):
        return PtensorX_TransferYFn.apply(self,_atoms, target_order)


    # ---- I/O -----------------------------------------------------------------------------------------------


    def __str__(self):
        u=_ptensor2.view(self,self.atoms)
        return u.__str__()

    def __repr__(self):
        u=_ptensor2.view(self,self.atoms)
        return u.__str__()



# ------------------------------------------------------------------------------------------------------------


class PtensorX_LinmapsYFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x, target_order: int):
        mult_factor = [
            [ 1, 1,  2],
            [ 1, 2,  5],
            [ 2, 5, 15],
        ][x.order,target_order]
        R = ptensorX.zeros(target_order,x.atoms,mult_factor*x.get_nc())
        u=_ptensorX(x.order).view(x,x.atoms)
        r=_ptensorX(target_order).view(R,R.atoms)
        [
            [ptens_base.add_linmaps0to0,ptens_base.add_linmaps0to1,ptens_base.add_linmaps0to2],
            [ptens_base.add_linmaps1to0,ptens_base.add_linmaps1to1,ptens_base.add_linmaps2to2],
            [ptens_base.add_linmaps2to0,ptens_base.add_linmaps2to1,ptens_base.add_linmaps2to2],
        ][x.order,target_order](r,u)
        ctx.mult_factor, ctx.order = mult_factor, x.order
        return R
        
    @staticmethod
    def backward(ctx,g):
        R = ptensorX.zeros(target_order,g.atoms,g.get_nc()/ctx.mult_factor)
        u=_ptensorX(g.order).view(g,g.atoms)
        r=_ptensorX(ctx.order).view(R,R.atoms)
        ptens_base.add_linmaps2to0_back(r,u)
        return R, None

class PtensorX_TransferYFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,_atoms, target_order: int):
        mult_factor = [
            [ 1, 1,  2],
            [ 1, 2,  5],
            [ 2, 5, 15],
        ][x.order,target_order]
        R = ptensorX.zeros(_atoms,x.get_nc()*mult_factor)
        u = _ptensorX(x.order).view(x,x.atoms)
        r = _ptensorX(target_order).view(R,R.atoms)
        ptens_base.add_msg(r,u)
        ctx.mult_factor, ctx.order = mult_factor, x.order
        return R
        
    @staticmethod
    def backward(ctx,g):
        R=ptensorX.zeros(g.atoms,g.get_nc()/ctx.mult_factor)
        u=_ptensorX(ctx.order).view(g,g.atoms)
        r=_ptensorX(g.order).view(R,R.atoms) 
        ptens_base.add_msg_back(r,u) 
        return R, None, None



