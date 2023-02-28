from typing import Callable, List, Optional, Tuple, Union
import torch
from ptens import ptensors0, ptensors1, ptensors2, graph
from ptens_base import atomspack
from ptens.functions import linear, linmaps0, outer, unite1, gather
######################################## Functions ###########################################
def ConvertEdgeAttributesToPtensors1(edge_index: torch.Tensor, edge_attributes: torch.Tensor):
  atoms = edge_index.transpose(0,1).float()
  return ptensors1.from_matrix(edge_attributes,atoms)

######################################## MODULES ###########################################
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

class LazyLinear(torch.nn.Module):
  def __init__(self,out_channels: int = None, bias: bool = True) -> None:
    r"""
    NOTE: if you do not initialize 'out_channels', it must be initialized before calling 'forward'.
    """
    super().__init__()
    #This follows Glorot initialization for weights.
    self.w = torch.nn.parameter.UninitializedParameter()
    self.b = torch.nn.parameter.UninitializedParameter() if bias else None
    self.out_channels = out_channels
  def reset_parameters(self):
    if not self.has_uninitialized_params():
      self.w = torch.nn.init.xavier_uniform_(self.w)
      if not self.b is None:
        self.b = torch.nn.init.zeros_(self.b)
  def initialize_parameters(self, input) -> None:
    if self.has_uninitialized_params():
      with torch.no_grad():
        self.weight.materialize((self.out_channels, input.get_nc()))
        if self.bias is not None:
            self.bias.materialize(self.out_channels)
        self.reset_parameters()
  def forward_standard(self,x: Union[ptensors0,ptensors1,ptensors2]) -> Union[ptensors0,ptensors1,ptensors2]:
    assert x.get_nc() == self.w.size(0)
    return x * self.w if self.b is None else linear(x,self.w,self.b)
  def forward(self,x: Union[ptensors0,ptensors1,ptensors2]) -> Union[ptensors0,ptensors1,ptensors2]:
    self.initialize_parameters(x)
    self.forward = self.forward_standard
    return self.forward(x)

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

# TODO: find a better place for the below function
def generate_generic_shape(name: str, size: int) -> graph:
  shape_types = ['star','cycle','path']
  assert name in shape_types
  assert size >= 0
  generate_star = lambda size: torch.tensor([[0,i,i,0] for i in range(1,size + 1)],dtype=torch.float32)
  generate_cycle = lambda size: torch.tensor([[i,(i + 1) % size,(i + 1) % size,i] for i in range(size)],dtype=torch.float32)
  generate_path = lambda size: torch.tensor([[i,i + 1,i + 1,i] for i in range(size - 1)],dtype=torch.float32)
  edge_index = [generate_star,generate_cycle,generate_path][shape_types.index(name)](size)
  edge_index = edge_index.view(-1,2).transpose(0,1)
  return graph.from_edge_index(edge_index,0)
  #return graph.from_matrix(torch.sparse_coo_tensor(edge_index,torch.ones(edge_index.size(1),dtype=torch.float)).to_dense())
def create_edge_ptensors1(edge_index: torch.Tensor, edge_attributes: torch.Tensor) -> ptensors1:
  atoms = edge_index.transpose(0,1).tolist()
  return ptensors1.from_matrix(edge_attributes,atoms)

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
class LazySubstructureTransferLayer(torch.nn.Module):
  def __init__(self, graph_filter: Union[graph,Tuple[str,int]], channels_out: int, out_order: int = None, bias : bool = True, reduction_type : str = "sum") -> None:
    r"""
    reduction_types: "sum" and "mean"
    leave 'out_order' as 'None' to keep same as input (NOTE: cannot leave as default if input is of order 0.)
    """
    super().__init__()
    assert reduction_type == "sum" or reduction_type == "mean"
    assert out_order in [None,1,2]
    if isinstance(graph_filter,Tuple[str,int]):
      graph_filter = generate_generic_shape(*graph_filter)
    assert isinstance(graph_filter,graph)
    self.lin = LazyLinear(channels_out,bias)
    self.use_mean = reduction_type == "mean"
    self.graph_filter = graph_filter
    self.out_order = out_order
    self.unite = None
  def reset_parameters(self):
    self.lin.reset_parameters()
  def forward(self, features: Union[ptensors0,ptensors1,ptensors2], graph: graph) -> Union[ptensors0,ptensors1,ptensors2]:
    domain_map = graph.overlaps(graph.subgraphs(self.graph_filter),atomspack(features.get_atoms()))
    if self.unite is None:
      if isinstance(features,ptensors0):
        in_order = 0
      elif isinstance(features,ptensors1):
        in_order = 1
      elif isinstance(features,ptensors2):
        in_order = 2
      else:
        raise "'features' must be instance of 'ptensors[0|1|2]'"
      out_order = in_order if self.out_order is None else self.out_order
      assert out_order is not None, "If 'in_order' is '0', then 'out_order' cannot be 'None'."
      #
      self.unite = [
        [ptensors0.unite1,ptensors0.unite2],
        [ptensors1.unite1,ptensors1.unite2],
        [ptensors2.unite1,ptensors2.unite2],
        ][in_order][out_order]
    F = self.unite(F,domain_map,self.use_mean)
    F = self.lin(F)
    return F
class LazySkipConnectionConvolutionalLayer(torch.nn.Module):
  def __init__(self, channels_out: int, reduction_type : str = "sum", out_order: int = None, bias : bool = True) -> None:
    r"""
    reduction_types: "sum" and "mean"
    leave 'out_order' as 'None' to keep same as input (NOTE: cannot leave as default if input is of order 0.)
    """
    super().__init__()
    assert reduction_type == "sum" or reduction_type == "mean"
    assert out_order in [None,0,1,2]
    self.lin1 = LazyLinear(channels_out,bias=False)
    self.lin2 = LazyLinear(channels_out,bias)
    self.use_mean = reduction_type == "mean"
    self.out_order = out_order
    self.transfer = None
  def reset_parameters(self):
    self.lin1.reset_parameters()
    self.lin2.reset_parameters()
  def forward(self, features: Union[ptensors0,ptensors1,ptensors2], graph: graph) -> Union[ptensors0,ptensors1,ptensors2]:
    if self.transfer is None:
      if isinstance(features,ptensors0):
        in_order = 0
      elif isinstance(features,ptensors1):
        in_order = 1
      elif isinstance(features,ptensors2):
        in_order = 2
      else:
        raise "'features' must be instance of 'ptensors[0|1|2]'"
      out_order = in_order if self.out_order is None else self.out_order
      assert out_order is not None, "If 'in_order' is '0', then 'out_order' cannot be 'None'."
      #
      self.lin1.out_channels = features.get_nc()
      #
      self.transfer = [
        [ptensors0.transfer1,ptensors0.transfer0],
        [ptensors1.transfer1,ptensors1.transfer0],
        [ptensors2.transfer1,ptensors2.transfer0],
        ][in_order][out_order]
      if in_order == 0:
        q = self.transfer
        self.transfer = lambda x,G,n: q(x,n)
    F1 = self.transfer(F,graph,F.get_atoms())
    F = self.lin2(F) + self.lin1(F1)
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