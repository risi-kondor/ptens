import torch

from ptens_base import ptensor2 as _ptensor2


class ptensor2(torch.Tensor):

    @classmethod
    def zeros(self, _atoms, _nc):
        R=ptensor2(torch.zeros(len(atoms),len(atoms),_nc))
        R.atoms=_atoms
        return R
    
    @classmethod
    def randn(self, _atoms, _nc):
        R=ptensor2(torch.randn(len(atoms),len(atoms),_nc))
        R.atoms=_atoms
        return R


    # ---- Access --------------------------------------------------------------------------------------------


    def get_nc(self):
        return self.size(2)


    # ---- Operations ----------------------------------------------------------------------------------------

    
    def linmaps0(self):
        return Ptensor2_Linmaps0Fn.apply(self);

    def linmaps1(self):
        return Ptensor2_Linmaps1Fn.apply(self);

    def linmaps2(self):
        return Ptensor2_Linmaps2Fn.apply(self);


    # ---- I/O -----------------------------------------------------------------------------------------------


    def __str__(self):
        u=_ptensor2.view(self,self.atoms)
        return u.__str__()



# ------------------------------------------------------------------------------------------------------------


class Ptensor2_Linmaps0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptensor0.zeros(x.atoms,2*x.get_nc())
        u=_ptensor2.view(x,x.atoms)
        u.add_linmaps_to(_ptensor0.view(R,R.atoms))
        return R
        
    @staticmethod
    def backward(ctx,g):
        print("ddp")


class Ptensor2_Linmaps0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptensor1.zeros(x.atoms,5*x.get_nc())
        u=_ptensor2.view(x,x.atoms)
        u.add_linmaps_to(_ptensor1.view(R,R.atoms))
        return R
        
    @staticmethod
    def backward(ctx,g):
        print("ddp")


class Ptensor2_Linmaps0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptensor2.zeros(x.atoms,15*x.get_nc())
        u=_ptensor2.view(x,x.atoms)
        _ptensor2.view(R,R.atoms).add_linmaps(u)
        return R
        
    @staticmethod
    def backward(ctx,g):
        print("ddp")


# ------------------------------------------------------------------------------------------------------------


def linmaps0(x):
    return x.linmaps0()
    
def linmaps1(x):
    return x.linmaps1()
    
def linmaps2(x):
    return x.linmaps2()

    
