import torch

from ptens_base import *

import ptens.ptensor0 
import ptens.ptensor1 


from ptens.ptensor0 import *
from ptens.ptensor1 import *
from ptens.ptensor2 import *

from ptens.ptensor0pack import *
#from ptens.ptensor1pack import *


class Ptensor0_Linmaps1Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptens.ptensor1.zeros(x.atoms,x.get_nc())
        u=_ptensor0.view(x,x.atoms)
        r=_ptensor1.view(R,R.atoms)
        ptens_base.add_linmaps0to1(r,u)
        return R
        
    @staticmethod
    def backward(ctx,g):
        R=ptensor0.zeros(g.atoms,g.get_nc())
        u=_ptensor1.view(g,g.atoms)
        r=_ptensor0.view(R,R.atoms)
        ptens_base.add_linmaps0to1_back(r,u) 
        return R


