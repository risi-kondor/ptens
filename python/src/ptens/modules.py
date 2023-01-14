from typing import Optional
import torch
from ptens import ptensors0, ptensors1, ptensors2, graph
from ptens.functions import linear, linmaps0, outer, unite1, gather
class Linear(torch.nn.Module):
  def __init__(self,in_channels: int, out_channels: int, bias: bool = True) -> None:
    super().__init__()
    #This follows Glorot initialization for weights.
    self.w = torch.nn.parameter.Parameter(torch.empty(in_channels,out_channels))
    self.b = torch.nn.parameter.Parameter(torch.empty(out_channels)) if bias else None
    self.reset_parameters()
  def reset_parameters(self):
    self.w = torch.nn.init.xavier_uniform_(self.w)
    if not self.b is None:
      self.b = torch.nn.init.zeros_(self.b)
  def forward(self,x: ptensors1) -> ptensors1:
    assert x.get_nc() == self.w.size(0)
    return x * self.w if self.b is None else linear(x,self.w,self.b) 

class ConvolutionalLayer_0P(torch.nn.Module):
  r"""
  An implementation of GCNConv in ptens.
  """
  def __init__(self, channels_in: int, channels_out: int, bias : bool = True) -> None:
    super().__init__()
    self.lin = Linear(channels_in,channels_out,bias)
  def reset_parameters(self):
    self.lin.reset_parameters()
  def forward(self, features: ptensors1, graph: graph, symm_norm: Optional[ptensors0] = None) -> ptensors1:
    r"""
    Give symm_norm if you want symmetric normalization.
    """
    if not symm_norm is None:
      features = outer(features,symm_norm)
    # VxVxk -> VxVxk
    F = gather(features,graph)
    # [VxVxk] [VxVxk*2] -> [VxVxk*3]
    # [VxV*k*2] -> [VxV*k']
    F = self.lin(F)
    if not symm_norm is None:
      features = outer(F,symm_norm)
    return F

class ConvolutionalLayer_1P(torch.nn.Module):
  def __init__(self, channels_in: int, channels_out: int, bias : bool = True, reduction_type : str = "sum") -> None:
    r"""
    reduction_types: "sum" and "mean"
    """
    super().__init__()
    assert reduction_type == "sum" or reduction_type == "mean"
    self.lin = Linear(2*channels_in,channels_out,bias)
    self.use_mean = reduction_type == "mean"
  def reset_parameters(self):
    self.lin.reset_parameters()
  def forward(self, features: ptensors1, graph: graph, symm_norm: Optional[ptensors0] = None) -> ptensors1:
    r"""
    Give symm_norm if you want symmetric normalization.
    """
    if not symm_norm is None:
      features = outer(features,symm_norm)
    # VxVxk -> VxVxk*2
    F = unite1(features,graph,self.use_mean)
    # [VxVxk] [VxVxk*2] -> [VxVxk*3]
    # [VxV*k*2] -> [VxV*k']
    F = self.lin(F)
    if not symm_norm is None:
      features = outer(F,symm_norm)
    return F
class Reduce_1P_0P(torch.nn.Module):
  def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
    super().__init__()
    self.lin = Linear(in_channels,out_channels,bias)
  def forward(self, features: ptensors1) -> ptensors0:
    features = linmaps0(features)
    a = self.lin(features)
    return a
class Dropout(torch.nn.Module):
  def __init__(self, prob: torch.float = 0.5, device : str = 'cuda') -> None:
    super().__init__()
    self.p = prob
    self.device = device
  def cuda(self, device = None):
    self.device = device
    return super().cuda(device)
  def forward(self, x):
    dropout = (torch.rand(x.get_nc(),device=self.device) > self.p).float()
    if isinstance(x,ptensors0):
      return ptensors0.mult_channels(x,dropout)
    elif isinstance(x,ptensors1):
      return ptensors1.mult_channels(x,dropout)
    elif isinstance(x,ptensors2):
      return ptensors2.mult_channels(x,dropout)
    else:
      raise NotImplementedError('Dropout not implemented for type \"' + str(type(x)) + "\"")
class BatchNorm(torch.nn.Module):
  def __init__(self, num_features: int, eps: torch.float = 1E-5, momentum: torch.float = 0.1) -> None:
    super().__init__()
    # TODO: consider using 'UnitializedParameter' here
    self.running_mean = None
    self.running_var = None
    self.eps = eps
    self.momentum = momentum
  def forward(self, x):
    r"""
    x can be any type of ptensors
    """
    x_vals : torch.Tensor = x.torch()
    if self.running_mean is None:
      self.has_had_first_batch = True
      self.running_mean = x_vals.mean(0)
      self.running_var = x_vals.var(0)
    else:
      self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_vals.mean()
      self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_vals.var(unbiased=False)
    y = (self.running_var + self.eps)**-0.5
    b = -self.running_mean * y
    test = torch.einsum('ij,jk->ij',x_vals,torch.diag(y)) + b
    print(test.mean(),test.var())
    # TODO: make this better
    return linear(x,torch.diag(y),b)
class PNormalize(torch.nn.Module):
  def __init__(self, p: int = 2, eps: torch.float = 1E-5) -> None:
    super().__init__()
    self.eps = eps
    self.p = p
  def forward(self, x):
    r"""
    x can be any type of ptensors
    """
    x_vals : torch.Tensor = x.torch()
    pnorm = torch.linalg.norm(x_vals,self.p,0)
    pnorm = (self.eps + pnorm)**-1
    if isinstance(x,ptensors0):
      return ptensors0.mult_channels(x,pnorm)
    elif isinstance(x,ptensors1):
      return ptensors1.mult_channels(x,pnorm)
    elif isinstance(x,ptensors2):
      return ptensors2.mult_channels(x,pnorm)
    else:
      raise NotImplementedError('PNormalize not implemented for type \"' + str(type(x)) + "\"")