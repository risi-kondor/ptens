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
import ptens
#import ptens.subgraph 

#import ptens.subgraphlayer0


def concat(x,y):
    return x.concat(y)

def linear(x,y,b):
    return x.linear(y,b)

def relu(x,alpha=0.5):
    return x.relu(alpha)


def linmaps0(x,normalized=False):
    return x.linmaps0(normalized)
    
def linmaps1(x,normalized=False):
    return x.linmaps1(normalized)
    
def linmaps2(x,normalized=False):
    return x.linmaps2(normalized)


def transfer0(x,_atoms,G,normalized=False):
    return x.transfer0(_atoms,G,normalized)
    
def transfer1(x,_atoms,G,normalized=False):
    return x.transfer1(_atoms,G,normalized)
    
def transfer2(x,_atoms,G,normalized=False):
    return x.transfer2(_atoms,G,normalized)

def unite1(x,G,normalized=False):
    return x.unite1(G,normalized)

def unite2(x,G,normalized=False):
    return x.unite2(G,normalized)


def gather(x,G):
    return x.gather(G)

def outer(x,y):
    return x.outer(y)


def device_id(device):
    if device==0:
        return 0
    if device==1:
        return 1
    if device=='cpu':
        return 0
    if device=='gpu':
        return 1
    if device=='cuda':
        return 1
    if device=='cuda:0':
        return 1
    return 0


class Linear(torch.nn.Module):
    
  def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
    super().__init__()
    #This follows Glorot initialization for weights.
    self.w = torch.nn.parameter.Parameter(torch.empty(in_channels,out_channels),requires_grad=True)
    self.b = torch.nn.parameter.Parameter(torch.empty(out_channels),requires_grad=True)
    self.reset_parameters()

  def reset_parameters(self):
    if not isinstance(self.w,torch.nn.parameter.UninitializedParameter):
      self.w = torch.nn.init.xavier_uniform_(self.w)
      if self.b is not None:
        self.b = torch.nn.init.zeros_(self.b)
  def forward(self,x):
    assert x.get_nc() == self.w.size(0), f'{x.get_nc()} != {self.w.size(0)}'
    #return x * self.w if self.b is None else linear(x,self.w,self.b)
    # TODO: figure out why multiplication is broken.
    return linear(x,self.w,torch.zeros(self.w.size(1),device=self.w.device) if self.b is None else self.b)

