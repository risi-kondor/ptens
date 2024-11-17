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


class batched_ptensorlayer0(p.batched_ptensorlayer):

    @classmethod
    def make(self,atoms,M):
        R=batched_ptensorlayer0(M)
        R.atoms=atoms
        return R

    @classmethod
    def zeros(self,atoms,nc,device='cpu'):
        assert isinstance(atoms,pb.batched_atomspack)
        assert isinstance(nc,int)
        return self.make(atoms,torch.zeros([atoms.nrows0(),nc],device=device))

    @classmethod
    def randn(self,atoms,nc,device='cpu'):
        assert isinstance(atoms,pb.batched_atomspack)
        assert isinstance(nc,int)
        return self.make(atoms,torch.randn([atoms.nrows0(),nc],device=device))

    @classmethod
    def from_ptensorlayers(self,list):
        for a in list:
            assert isinstance(a,p.ptensorlayer)
        atoms=pb.batched_atomspack([a.atoms for a in list])
        M=torch.cat(list,0)
        return self.make(atoms,M)

    @classmethod
    def from_matrix(self,atoms,M):
        assert isinstance(atoms,pb.batched_atomspack)
        assert isinstance(M,torch.Tensor)
        assert M.dim()==2
        assert M.size(0)==atoms.nrows0()
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
        return batched_ptensorlayer0.make(atoms,M)
    
    def zeros_like(self):
        return batched_ptensorlayer0.zeros(self.atoms,self.get_nc(),device=self.device)
    
    def backend(self):
        return pb.batched_ptensors0.view(self.atoms,self)


    # ----- Access -------------------------------------------------------------------------------------------


    def getk(self):
        return 0
    
    def __len__(self):
        return len(self.atoms)
    
    def get_nc(self):
        return self.size(1)
    
    def __getitem__(self,i):
        assert i<len(self)
        offs=self.atoms.offset0(i)
        n=self.atoms.nrows0(i)
        M=torch.Tensor(self)[offs:offs+n,:]
        return p.ptensorlayer0.from_matrix(self.atoms[i],M)


    # ---- Linmaps -------------------------------------------------------------------------------------------
    

    @classmethod
    def linmaps(self,x):
        return batched_ptensorlayer0_linmapsFn.apply(x)


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
        return batched_ptensorlayer0_gatherFn.apply(atoms,x,map)
        

    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return "batched_ptensorlayer0(size="+str(len(self))+",nc="+str(self.size(1))+")"

    def __str__(self,indent=""):
        r=""
        r=r+indent+self.__repr__()+":\n"
        for i in range(len(self)):
            r=r+self[i].__str__("  ")+"\n"
        return r



# ---- Autograd functions --------------------------------------------------------------------------------------------


# class batched_ptensorlayer0_catFn(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx,x):
#         nb=len(x[0])
#         ncat=len(x)
#         offsets=torch.zeros([nb,ncat])
#         for i in range(0,nb):
#             for j in range(0,ncat):
#                 offsets[i,j]=x[j].atoms.nrows0(i)
#         for i in range(0,nb):
#             offs=R.atoms.offset0(i)
#             for j in range(0,ncat):
#                 xoffs=x[j].atoms.offset0(i)
#                 R[offs,offs+offsets[i,j],:]=x[j][xoffs:xoffs+offsets[i,j],:]
#                 xoffs=xoffs+offsets[i,j]
#         ctx.atomsv=[p.atoms for p in x]
#         ctx.nrowsv=[p.size(0) for p in x]
#         return ptensorlayer0.make(pb.atomspack.cat(ctx.atomsv),torch.cat(x))

#     @staticmethod
#     def backward(ctx,g):
#         r=[]
#         offs=0
#         for p in zip(ctx.atomsv,ctx.nrowsv):
#             r.append(ptensorlayer0.make(p[0],g[offs:offs+p[1],:]))
#             offs=offs+p[1]
#         return tuple(r)


class batched_ptensorlayer0_linmapsFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        r=batched_ptensorlayer0.zeros(x.atoms,x.get_nc()*([1,1,2][x.getk()]),device=x.device)
        r.backend().add_linmaps(x.backend())
        ctx.x=x
        return r

    @staticmethod
    def backward(ctx,g):
        r=ctx.x.zeros_like()
        r.backend().add_linmaps_back(g.backend())
        return r


class batched_ptensorlayer0_gatherFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,atoms,x,map):
        r=batched_ptensorlayer0.zeros(atoms,x.get_nc()*([1,1,2][x.getk()]),device=x.device)
        r.backend().add_gather(x.backend(),map)
        return r

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        atoms, x, map = inputs
        ctx.map = map
        ctx.atoms = atoms
        ctx.save_for_backward(x)
    
    @staticmethod
    def backward(ctx,g):
        x, = ctx.saved_tensors
        r = x.zeros_like()
        g_view = pb.batched_ptensors0.view(ctx.atoms, g)
        r.backend().add_gather_back(g_view, ctx.map)
        return None, r, None



