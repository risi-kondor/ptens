#
# This file is part of ptens, a C++/CUDA library for permutation 
# equivariant message passing. 
#  
# Copyright (c) 2024, Imre Risi Kondor
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
import ptens.ptensor0 as ptensor0


class batched_ptensorlayer2(p.batched_ptensorlayer):

    @classmethod
    def make(self,atoms,M):
        R=batched_ptensorlayer2(M)
        R.atoms=atoms
        return R

    @classmethod
    def zeros(self,atoms,nc,device='cpu'):
        assert isinstance(atoms,pb.batched_atomspack)
        assert isinstance(nc,int)
        return self.make(atoms,torch.zeros([atoms.nrows2(),nc],device=device))

    @classmethod
    def randn(self,atoms,nc,device='cpu'):
        assert isinstance(atoms,pb.batched_atomspack)
        assert isinstance(nc,int)
        return self.make(atoms,torch.randn([atoms.nrows2(),nc],device=device))

    @classmethod
    def from_ptensorlayers(self,list):
        for a in list:
            assert isinstance(a,p.ptensorlayer2)
        atoms=pb.batched_atomspack([a.atoms for a in list])
        return self.make(atoms,torch.cat(list,0))

    @classmethod
    def from_matrix(self,atoms,M):
        assert isinstance(atoms,pb.batched_atomspack)
        assert isinstance(M,torch.Tensor)
        assert M.dim()==2
        assert M.size(0)==atoms.nrows2()
        return self.make(atoms,M)

    @classmethod
    def cat(self,x):
        nb=len(x[0])
        ncat=len(x)
        atoms=pb.batched_atomspack.cat([p.atoms for p in x])
        M=torch.zeros([atoms,torch.sum(torch.tensor([p.size() for p in x]))],device=x[0].device)
        for i in range(0,nb):
            offs=R.atoms.offset0(i)
            for j in range(0,ncat):
                xoffs=x[j].atoms.offset0(i)
                nrows=x[j].atoms.nrows0(i)
                M[offs,offs+nrows,:]=x[j][xoffs:xoffs+nrows,:]
                offs+=nrows
        return batched_ptensorlayer2.make(atoms,M)
    
    def zeros_like(self):
        return batched_ptensorlayer2.zeros(self.atoms,self.get_nc(),device=self.device)
    
    def backend(self):
        return pb.batched_ptensors2.view(self.atoms,self)


    # ----- Access -------------------------------------------------------------------------------------------


    def getk(self):
        return 2
    
    def __len__(self):
        return len(self.atoms)
    
    def get_nc(self):
        return self.size(1)
    
    def __getitem__(self,i):
        assert i<len(self)
        offs=self.atoms.offset2(i)
        n=self.atoms.nrows2(i)
        M=torch.Tensor(self)[offs:offs+n,:]
        return p.ptensorlayer2.from_matrix(self.atoms[i],M)


    # ---- Linmaps -------------------------------------------------------------------------------------------
    

    @classmethod
    def linmaps(self,x):
        return batched_ptensorlayer2_linmapsFn.apply(x)


    # ---- Message passing -----------------------------------------------------------------------------------


    @classmethod
    def gather(self,atoms,x,*args):
        assert isinstance(atoms,pb.batched_atomspack)
        assert isinstance(x,p.batched_ptensorlayer)
        if len(args)==0:
            map=pb.batched_layer_map.overlaps_map(atoms,x.atoms)
        else:
            map=args[0]
        assert isinstance(map,pb.batched_layer_map)
        return batched_ptensorlayer2_gatherFn.apply(atoms,x,map)
        

    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return "batched_ptensorlayer2(size="+str(len(self))+",nc="+str(self.size(1))+")"

    def __str__(self,indent=""):
        r=""
        r=r+indent+self.__repr__()+":\n"
        for i in range(len(self)):
            r=r+self[i].__str__("  ")+"\n"
        return r



# ---- Autograd functions --------------------------------------------------------------------------------------------


class batched_ptensorlayer2_linmapsFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        r=batched_ptensorlayer2.zeros(x.atoms,x.get_nc()*([2,5,15][x.getk()]),device=x.device)
        r.backend().add_linmaps(x.backend())
        ctx.x=x
        return r

    @staticmethod
    def backward(ctx,g):
        r=ctx.x.zeros_like()
        r.backend().add_linmaps_back(g.backend())
        return r


class batched_ptensorlayer2_gatherFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,atoms,x,tmap):
        r=batched_ptensorlayer2.zeros(atoms,x.get_nc()*([2,5,15][x.getk()]),device=x.device)
        r.backend().add_gather(x.backend(),tmap)
        ctx.save_for_backward(x)
        ctx.tmap=tmap
        ctx.atoms = atoms
        return r

    @staticmethod
    def backward(ctx,g):
        x, = ctx.saved_tensors
        r = x.zeros_like()
        g_view = pb.batched_ptensors2.view(ctx.atoms, g)
        r.backend().add_gather_back(g_view, ctx.tmap)
        return None, r, None



