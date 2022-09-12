import torch

import ptens_base 
from ptens_base import ptensor0 as _ptensor0
from ptens_base import ptensor1 as _ptensor1
from ptens_base import ptensor2 as _ptensor2

#from ptens import ptensor0 as ptensor0
import ptens.ptensor0 
import ptens.ptensor2 

class ptensor1(torch.Tensor):

    @classmethod
    def zeros(self, _atoms, _nc):
        R=ptensor1(torch.zeros(len(_atoms),_nc))
        R.atoms=_atoms
        return R
    
    @classmethod
    def randn(self, _atoms, _nc):
        R=ptensor1(torch.randn(len(_atoms),_nc))
        R.atoms=_atoms
        return R

    @classmethod
    def sequential(self, _atoms, _nc):
        R=ptensor1(_ptensor1.sequential(_atoms,_nc).torch())
        R.atoms=_atoms
        return R


    # ---- Access --------------------------------------------------------------------------------------------


    def get_nc(self):
        return self.size(1)


    # ---- Operations ----------------------------------------------------------------------------------------

    
    def linmaps0(self):
        return Ptensor1_Linmaps0Fn.apply(self);

    def linmaps1(self):
        return Ptensor1_Linmaps1Fn.apply(self);

    def linmaps2(self):
        return Ptensor1_Linmaps2Fn.apply(self);


    # ---- I/O -----------------------------------------------------------------------------------------------


    def __str__(self):
        u=_ptensor1.view(self,self.atoms)
        return u.__str__()



# ------------------------------------------------------------------------------------------------------------


class Ptensor1_Linmaps0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptens.ptensor0.zeros(x.atoms,x.get_nc())
        u=_ptensor1.view(x,x.atoms)
        r=_ptensor0.view(R,R.atoms)
        ptens_base.add_linmaps1to0(r,u)
        return R
        
    @staticmethod
    def backward(ctx,g):
        R=ptensor1.zeros(g.atoms,g.get_nc())
        u=_ptensor0.view(g,g.atoms)
        r=_ptensor1.view(R,R.atoms)
        ptens_base.add_linmaps1to0_back(r,u)
        return R


class Ptensor1_Linmaps1Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptensor1.zeros(x.atoms,2*x.get_nc())
        u=_ptensor1.view(x,x.atoms)
        r=_ptensor1.view(R,R.atoms)
        ptens_base.add_linmaps1to1(r,u)
        return R
        
    @staticmethod
    def backward(ctx,g):
        R=ptensor1.zeros(g.atoms,g.get_nc()/2)
        u=_ptensor1.view(g,g.atoms)
        r=_ptensor1.view(R,R.atoms)
        ptens_base.add_linmaps1to1_back(r,u)
        return R


class Ptensor1_Linmaps2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptens.ptensor2.zeros(x.atoms,5*x.get_nc())
        u=_ptensor1.view(x,x.atoms)
        r=_ptensor2.view(R,R.atoms)
        ptens_base.add_linmaps1to2(r,u)
        return R
        
    @staticmethod
    def backward(ctx,g):
        R=ptensor1.zeros(g.atoms,g.get_nc()/2)
        u=_ptensor2.view(g,g.atoms)
        r=_ptensor1.view(R,R.atoms)
        ptens_base.add_linmaps1to2_back(r,u)
        return R


# ------------------------------------------------------------------------------------------------------------


def linmaps0(x):
    return x.linmaps0()

def linmaps1(x):
    return x.linmaps1()
    
def linmaps2(x):
    return x.linmaps2()
    

        
