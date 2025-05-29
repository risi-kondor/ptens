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

import ptens_base as pb 
import ptens as p 
import ptens.ptensorlayer as ptensorlayer
import ptens.ptensor0 as ptensor0


class ptensorlayer0(ptensorlayer):

    @classmethod
    def make(self,atoms,M):
        if isinstance(atoms,list):
           atoms=pb.atomspack(atoms)
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(M,torch.Tensor)
        M=M.to(dtype=torch.float32)
        R=ptensorlayer0(M)
        R.atoms=atoms
        return R

    @classmethod
    def zeros(self,atoms,nc,device='cpu'):
        if isinstance(atoms,list):
           atoms=pb.atomspack(atoms)
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(nc,int)
        return self.make(atoms,torch.zeros([len(atoms),nc],device=device))

    @classmethod
    def randn(self,atoms,nc,device='cpu'):
        if isinstance(atoms,list):
           atoms=pb.atomspack(atoms)
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(nc,int)
        return self.make(atoms,torch.randn([len(atoms),nc],device=device))

    @classmethod
    def sequential(self,atoms,nc,device='cpu'):
        if isinstance(atoms,list):
           atoms=pb.atomspack(atoms)
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(nc,int)
        return self.make(atoms,torch.tensor([i for i in range (0,atoms.nrows0()*nc)],dtype=torch.float,device=device).reshape(atoms.nrows0(),nc))

    @classmethod
    def from_matrix(self,atoms,M):
        if isinstance(atoms,list):
           atoms=pb.atomspack(atoms)
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(M,torch.Tensor)
        assert M.dim()==2
        assert M.size(0)==atoms.nrows0()
        return self.make(atoms,M)

    @classmethod
    def cat(self,args):
        return ptensorlayer0_catFn.apply(args)
    
    def zeros_like(self):
        return ptensorlayer0.zeros(self.atoms,self.get_nc(),device=self.device)
    
    def backend(self):
        return pb.ptensors0.view(self.atoms,self)


    # ----- Access -------------------------------------------------------------------------------------------


    def getk(self):
        return 0
    
    def __len__(self):
        return len(self.atoms)
    
    def get_nc(self):
        return self.size(1)
    
    def __getitem__(self,i):
        assert i<len(self)
        return ptensor0.from_tensor(self.atoms[i],torch.Tensor(self)[i])


    # ---- Linmaps -------------------------------------------------------------------------------------------
    

    @classmethod
    def linmaps(self,x):
        return ptensorlayer0_linmapsFn.apply(x)
    

    # ---- Message passing -----------------------------------------------------------------------------------


    @classmethod
    def gather(self,atoms,x,*args):
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(x,p.ptensorlayer)
        if len(args)==0:
            map=pb.layer_map.overlaps_map(atoms,x.atoms)
        else:
            map=args[0]
        return ptensorlayer0_gatherFn.apply(atoms,x,map)
        

    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return "ptensorlayer0(len="+str(self.size(0))+",nc="+str(self.size(1))+")"

    def __str__(self,indent=""):
        r=indent+"Ptensorlayer0:\n"
        r=r+self.backend().__str__(indent+"  ")
        return r



# ---- Autograd functions --------------------------------------------------------------------------------------------


class ptensorlayer0_catFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        ctx.atomsv=[p.atoms for p in x]
        ctx.nrowsv=[p.size(0) for p in x]
        return ptensorlayer0.make(pb.atomspack.cat(ctx.atomsv),torch.cat(x))

    @staticmethod
    def backward(ctx,g):
        r=[]
        offs=0
        for p in zip(ctx.atomsv,ctx.nrowsv):
            r.append(ptensorlayer0.make(p[0],g[offs:offs+p[1],:]))
            offs=offs+p[1]
        return tuple(r)


class ptensorlayer0_linmapsFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        r=ptensorlayer0.zeros(x.atoms,x.get_nc()*([1,1,2][x.getk()]),device=x.device)
        r.backend().add_linmaps(x.backend())
        ctx.atoms = x.atoms
        ctx.save_for_backward(x)
        return r
            
    @staticmethod
    def backward(ctx,g):
        x, = ctx.saved_tensors
        r = x.zeros_like()
        g_view = pb.ptensors0.view(ctx.atoms, g)
        r.backend().add_linmaps_back(g_view)
        return r


class ptensorlayer0_gatherFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, atoms,x,map):
        r=ptensorlayer0.zeros(atoms,x.get_nc()*([1,1,2][x.getk()]),device=x.device)
        r.backend().add_gather(x.backend(),map)
        ctx.save_for_backward(x)
        ctx.atoms = atoms
        ctx.map = map
        return r
  
    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        r=x.zeros_like()
        g_view = pb.ptensors0.view(ctx.atoms, g)
        r.backend().add_gather_back(g_view, ctx.map)
        #      atoms,x, map
        return None, r, None




