from typing import List, Literal, Optional, Tuple, Union
from typing_extensions import override
import torch
import torch.nn as nn
from torch.nn import LeakyReLU
from torch.nn.parameter import Parameter
from torch.nn.functional import softmax, dropout
import ptens 
from ptens import ptensors0, ptensors1, ptensors2, graph
from warnings import warn
from ptens_base import atomspack
from ptens.functions import linear, linmaps0, outer, unite1, unite2, gather, relu #, cat
######################################## Functions ###########################################


def get_edge_maps(edge_index: torch.Tensor, num_nodes: Optional[int] = None) -> Tuple[ptens.graph,ptens.graph]:
  if num_nodes is None:
    num_nodes = int(edge_index.max().item()) + 1
  edge_ids = torch.arange(edge_index.size(1),dtype=torch.float)
  E_to_G = ptens.graph.from_edge_index(torch.stack([edge_index[1],edge_ids]),n=num_nodes,m=edge_index.size(1))
  G_to_E = ptens.graph.from_edge_index(torch.stack([edge_ids,edge_index[0]]),n=edge_index.size(1),m=num_nodes)
  return E_to_G, G_to_E


class MapInfo:
  r"""
    Boiler Plate class for storing information regarding a map between subgraphs.
  """
  def __init__(self, forward_map: graph, backward_map: Optional[graph], source_domains: Optional[atomspack], target_domains: Optional[atomspack], subgraph: Optional[graph]):
    self.forward_map = forward_map
    self.backward_map = backward_map
    self.source_domains = source_domains
    self.target_domains = target_domains
    self.subgraph = subgraph
def ComputeSubstructureMap(source_domains: atomspack, graph_filter: graph, G: graph, extra_info: bool = False, include_subgraph_graph: bool = False) -> Union[MapInfo,None]:
  r"""
  Returns None if subgraph does not appear
  """
  # TODO: make it so .get_atoms() returns an atomspack, so we don't have to convert back to one.
  subgraphs = G.subgraphs(graph_filter)
  if len(subgraphs) == 0:
    return None
  info = [graph.overlaps(subgraphs,source_domains)]
  if extra_info:
    info.extend([
      graph.overlaps(source_domains,subgraphs),
      source_domains,
      subgraphs,
    ])
  info = MapInfo(*info,subgraph=graph.overlaps(subgraphs,subgraphs) if include_subgraph_graph else None)
  return info

######################################## MODULES ###########################################

def reset_params_recursive(module: torch.nn.Module):
  if hasattr(module,'reset_parameters'):
    module.reset_parameters()
  else:
    assert len([p for p in module.parameters(False) if p.requires_grad]) == 0
    for child in  module.children():
      reset_params_recursive(child)  

class GINConv_0P(torch.nn.Module):
  def __init__(self, nn: torch.nn.Module, eps: float = 0, train_eps: bool = True, reduction_type: Union[Literal['mean'],Literal['sum']] = 'sum') -> None:
    super().__init__()
    self.nn = nn
    eps += 1
    eps = [eps]
    self.eps_shifted_init = eps
    if train_eps:
      self.eps = torch.nn.parameter.Parameter(torch.tensor(eps,dtype=torch.float),requires_grad=True)
    else:
      self.register_buffer('eps',torch.tensor(eps,dtype=torch.float))
    self.normalize_reduction = reduction_type == 'mean'
  def reset_parameters(self):
    reset_params_recursive(self.nn)
    if self.eps.requires_grad:
      self.eps.data = torch.tensor(self.eps_shifted_init,dtype=torch.float,device=self.eps.device,requires_grad=True)
  def forward(self, x: ptens.ptensors0, G: ptens.graph):
    #x = self.eps * x + x.gather(G,self.normalize_reduction)
    x = x.mult_channels(self.eps *torch.ones(x.get_nc(),device=self.eps.device)) + x.gather(G,self.normalize_reduction)
    return self.nn(x)


class GINEConv_0P(torch.nn.Module):
  def __init__(self, nn: torch.nn.Module, eps: float = 0, train_eps: bool = True, reduction_type: Union[Literal['mean'],Literal['sum']] = 'sum') -> None:
    super().__init__()
    self.nn = nn
    eps += 1
    eps = [eps]
    self.eps_shifted_init = eps
    if train_eps:
      self.eps = torch.nn.parameter.Parameter(torch.tensor(eps,dtype=torch.float),requires_grad=True)
    else:
      self.register_buffer('eps',torch.tensor(eps,dtype=torch.float))
    self.normalize_reduction = reduction_type == 'mean'
  def reset_parameters(self):
    reset_params_recursive(self.nn)
    if self.eps.requires_grad:
      self.eps.data = torch.tensor(self.eps_shifted_init,dtype=torch.float,device=self.eps.device,requires_grad=True)
  def forward(self, x: ptens.ptensors0, e: ptens.ptensors0, vertex_to_edge_map: ptens.graph, edge_to_vertex_map: ptens.graph):
    y : ptens.ptensors0 = x.gather(vertex_to_edge_map)
    y = y + e
    y = y.relu(0)
    y = y.gather(edge_to_vertex_map,self.normalize_reduction)
    x = ptens.ptensors0.mult_channels(x,self.eps *torch.ones(x.get_nc(),device=self.eps.device))
    x = x + y
    return self.nn(x)

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
  def forward(self,x: Union[ptensors0,ptensors1,ptensors2]) -> Union[ptensors0,ptensors1,ptensors2]:
    assert x.get_nc() == self.w.size(0), f'{x.get_nc()} != {self.w.size(0)}'
    #return x * self.w if self.b is None else linear(x,self.w,self.b)
    # TODO: figure out why multiplication is broken.
    return linear(x,self.w,torch.zeros(self.w.size(1),device=self.w.device) if self.b is None else self.b)

class LazyLinear(torch.nn.Module):
  def __init__(self,out_channels: Optional[int] = None, bias: bool = True) -> None:
    r"""
    NOTE: if you do not initialize 'out_channels', it must be initialized before calling 'forward'.
    """
    super().__init__()
    #This follows Glorot initialization for weights.
    self.w = torch.nn.parameter.UninitializedParameter()
    self.b = torch.nn.parameter.UninitializedParameter() if bias else None
    self.out_channels = out_channels
  def reset_parameters(self):
    if not isinstance(self.w,torch.nn.parameter.UninitializedParameter):
      self.w = torch.nn.init.xavier_uniform_(self.w)
      if self.b is not None:
        self.b = torch.nn.init.zeros_(self.b)
  def initialize_parameters(self, input) -> None:
    if isinstance(self.w,torch.nn.parameter.UninitializedParameter):
      assert self.out_channels is not None, "Cannot initialize parameters for LazyLinear layer when 'out_channels' is None."
      with torch.no_grad():
        self.w.materialize((input.get_nc(),self.out_channels))
        if self.b is not None:
          self.b.materialize(self.out_channels)# type: ignore
        self.reset_parameters()
  def forward_standard(self,x: Union[ptensors0,ptensors1,ptensors2]) -> Union[ptensors0,ptensors1,ptensors2]:
    assert x.get_nc() == self.w.size(0)
    #return x * self.w if self.b is None else linear(x,self.w,self.b)
    # TODO: figure out why multiplication is broken.
    return linear(x,self.w,torch.zeros(self.w.size(1),device=self.w.device) if self.b is None else self.b)
  def forward(self,x: Union[ptensors0,ptensors1,ptensors2]) -> Union[ptensors0,ptensors1,ptensors2]:
    self.initialize_parameters(x)
    self.forward = self.forward_standard
    return self.forward(x)

class ConvolutionalLayer_0P(torch.nn.Module):
  r"""
  An implementation of GCN in ptens.
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
      features = outer(features,symm_norm) # type: ignore
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
  return graph.from_edge_index(edge_index)
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
      features = outer(features,symm_norm) # type: ignore
    # VxVxk -> VxVxk*2
    F = unite1(features,graph,self.use_mean)
    # [VxVxk] [VxVxk*2] -> [VxVxk*3]
    # [VxV*k*2] -> [VxV*k']
    F = self.lin(F)
    if not symm_norm is None:
      features = outer(F,symm_norm)
    return F
def _get_mult_factor(in_order: Literal[0,1,2], out_order: Literal[0,1,2]) -> int:
  return [
    [1,1,1],
    [1,2,5],
    [1,5,15],
  ][in_order][out_order]

class Unite(torch.nn.Module):
  def __init__(self, channels_in: int, channels_out: int, in_order: Literal[0,1,2], out_order: Literal[0,1,2], bias : bool = True, reduction_type : Literal['sum','mean'] = 'sum') -> None:
    r"""
    reduction_types: "sum" and "mean"
    leave 'out_order' and 'channels_out' as 'None' to keep same as input
    """
    super().__init__()
    self.lin = Linear(channels_in*_get_mult_factor(in_order,out_order),channels_out,bias)
    self.use_mean = reduction_type == "mean"
    self.unite = [lambda x, G, n: linmaps0(x,n),unite1,unite2][out_order]

  def reset_parameters(self):
    self.lin.reset_parameters()
  def forward(self, features: Union[ptensors0,ptensors1,ptensors2], graph: graph) -> Union[ptensors0,ptensors1,ptensors2]:
    F = self.unite(features,graph,self.use_mean)
    F = self.lin(F)
    return F
class LazyUnite(torch.nn.Module):
  def __init__(self, channels_out: Optional[int] = None, out_order: Optional[int] = None, bias : bool = True, reduction_type : Literal['sum','mean'] = 'sum') -> None:
    r"""
    reduction_types: "sum" and "mean"
    leave 'out_order' and 'channels_out' as 'None' to keep same as input
    """
    super().__init__()
    assert reduction_type == "sum" or reduction_type == "mean"
    assert out_order in [None,1,2]
    self.lin = LazyLinear(channels_out,bias)
    self.use_mean = reduction_type == "mean"
    self.out_order = out_order
    self.unite = None
  def reset_parameters(self):
    self.lin.reset_parameters()
  def forward(self, features: Union[ptensors0,ptensors1,ptensors2], graph: graph) -> Union[ptensors0,ptensors1,ptensors2]:
    if self.unite is None:
      if self.lin.out_channels is None:
        self.lin.out_channels = features.get_nc()
      if isinstance(features,ptensors0):
        in_order = 0
      elif isinstance(features,ptensors1):
        in_order = 1
      elif isinstance(features,ptensors2):
        in_order = 2
      else:
        raise Exception("'features' must be instance of 'ptensors[0|1|2]'")
      out_order = in_order if self.out_order is None else self.out_order
      assert out_order is not None, "If 'in_order' is '0', then 'out_order' cannot be 'None'."
      #
      #self.unite = [
      #  [lambda x,y: x,ptensors0.unite1,ptensors0.unite2],
      #  [ptensors1.linmaps0,ptensors1.unite1,ptensors1.unite2],
      #  [ptensors2.linmaps0,ptensors2.unite1,ptensors2.unite2],
      #  ][in_order][out_order]
      self.unite = [
        linmaps0,unite1,unite2
      ][out_order]
      if in_order == 0 and out_order == 0:
        warn("You have 'in_order = out_order = 0', this means the unite layer is simply a linear layer.  Did you intend this?")
      if out_order == 0:
        q = self.unite
        self.unite = lambda x,G,n: q(x,n)
    F = self.unite(features,graph,self.use_mean)
    F = self.lin(F)
    return F
class LazySubstructureTransport(LazyUnite):
  def __init__(self, channels_out: int, graph_filter: Union[graph,Tuple[str,int],None] = None, out_order: Optional[int] = None, bias : bool = True, reduction_type : str = "sum") -> None:
    r"""
    reduction_types: "sum" and "mean"
    leave 'out_order' as 'None' to keep same as input (NOTE: cannot leave as default if input is of order 0.)
    """
    super().__init__(channels_out,out_order,bias,reduction_type)
    assert out_order != 0
    if isinstance(graph_filter,tuple):
      graph_filter = generate_generic_shape(*graph_filter)
    assert isinstance(graph_filter,graph)
    self.graph_filter = graph_filter
  def forward(self, features: Union[ptensors0,ptensors1,ptensors2], graph: graph) -> Union[ptensors0,ptensors1,ptensors2,None]:
    r"""
    Returns 'None' if nothing is mapped to.
    """
    domain_map = ComputeSubstructureMap(features.get_atoms(),self.graph_filter,graph)
    if domain_map is None:
      return None
    return super().forward(features,domain_map.forward_map) # type: ignore
class Transfer(torch.nn.Module):
  def __init__(self, in_channels: int, channels_out: int, in_order: Literal[0,1,2], out_order: Literal[0,1,2], reduction_type : Literal['mean','sum'] = "sum", bias : bool = True) -> None:
    super().__init__()
    self.lin = Linear(in_channels * _get_mult_factor(in_order,out_order),channels_out,bias)
    self.use_mean = reduction_type == "mean"
    self.transfer = [ptens.transfer0,ptens.transfer1,ptens.transfer2][out_order]
  def reset_parameters(self):
    self.lin.reset_parameters()
  def forward(self, features: Union[ptensors0,ptensors1,ptensors2], target_domains: Union[List[List[int]],atomspack], graph: graph) -> Union[ptensors0,ptensors1,ptensors2]:
    F = self.transfer(features,target_domains, graph, self.use_mean)
    F = self.lin(F)
    return F
class LazyTransfer(torch.nn.Module):
  def __init__(self, channels_out: Optional[int] = None, reduction_type : str = "sum", out_order: Optional[int] = None, bias : bool = True) -> None:
    r"""
    reduction_types: "sum" and "mean"
    leave 'out_order' and 'channels_out' as 'None' to keep same as input
    """
    super().__init__()
    assert reduction_type == "sum" or reduction_type == "mean"
    assert out_order in [None,0,1,2]
    self.lin = LazyLinear(channels_out,bias)
    self.use_mean = reduction_type == "mean"
    self.out_order = out_order
    self.transfer = None
  def forward(self, features: Union[ptensors0,ptensors1,ptensors2], target_domains: Union[List[List[int]],atomspack], graph: graph) -> Union[ptensors0,ptensors1,ptensors2]:
    if self.transfer is None:
      if self.lin.out_channels is None:
        self.lin.out_channels = features.get_nc()
      if isinstance(features,ptensors0):
        in_order = 0
      elif isinstance(features,ptensors1):
        in_order = 1
      elif isinstance(features,ptensors2):
        in_order = 2
      else:
        raise Exception("'features' must be instance of 'ptensors[0|1|2]'")
      out_order = in_order if self.out_order is None else self.out_order
      #
      self.transfer = [
        [ptensors0.transfer0,ptensors0.transfer1,ptensors0.transfer2],
        [ptensors1.transfer0,ptensors1.transfer1,ptensors1.transfer2],
        [ptensors2.transfer0,ptensors2.transfer1,ptensors2.transfer2],
        ][in_order][out_order]
      if in_order == 0:
        q = self.transfer
        self.transfer = lambda x,t,G,n: q(x,G,n)
    F = self.transfer(features,target_domains, graph, self.use_mean)
    F = self.lin(F)
    return F
class LazyTransferNHoods(LazyTransfer):
  def __init__(self, num_hops: int, channels_out: Optional[int] = None, reduction_type : str = "sum", out_order: Optional[int] = None, bias : bool = True) -> None:
    r"""
    reduction_types: "sum" and "mean"
    leave 'out_order' as 'None' to keep same as input
    """
    super().__init__(channels_out,reduction_type,out_order,bias)
    self.num_hops = num_hops
  def forward(self, features: Union[ptensors0,ptensors1,ptensors2], graph: graph) -> Union[ptensors0,ptensors1,ptensors2]:
    return super().forward(features,graph.nhoods(self.num_hops),graph)
class LazyPtensGraphConvolutional(LazyTransfer):
  def __init__(self, channels_out: Optional[int] = None, reduction_type : str = "sum", out_order: Optional[int] = None, bias : bool = True) -> None:
    r"""
    This is a generalization of GCN to higher orders.  The limiting factor is that the source and target domains must be the same.
    reduction_types: "sum" and "mean"
    leave 'out_order' and 'channels_out' as 'None' to keep same as input (NOTE: cannot leave as default if input is of order 0.)
    """
    super().__init__(channels_out,reduction_type,out_order,bias)
    self.lin2 = LazyLinear(channels_out,bias=False)
  def forward_adv(self, features: Union[ptensors0,ptensors1,ptensors2], graph: graph) -> Union[ptensors0,ptensors1,ptensors2]:
    return super().forward(features,features.get_atoms(),graph) + self.lin2(features)
  def forward(self, features: Union[ptensors0,ptensors1,ptensors2], graph: graph) -> Union[ptensors0,ptensors1,ptensors2]:
    self.forward = self.forward_adv
    if self.lin2.out_channels is None:
      self.lin2.out_channels = features.get_nc()
    return self.forward(features,graph)
class Dropout(torch.nn.Module):
  def __init__(self, prob: float = 0.5) -> None: # type: ignore
    super().__init__()
    self.p = prob
    self.device_holder = torch.nn.parameter.Parameter()
  def forward(self, x):
    if self.p == 0:
      return x
    # TODO: replace device with device from 'x'.
    if self.training:
      dropout = 1/(1 - self.p)*(torch.rand(x.get_nc(),device=self.device_holder.device) > self.p) # type: ignore
      return x.mult_channels(dropout)
    else:
      return x
class BatchNorm(torch.nn.BatchNorm1d):
    @override
    def forward(self, input: ptens.ptensors2) -> ptens.ptensors2:...
    @override
    def forward(self, input: ptens.ptensors1) -> ptens.ptensors1:...
    @override
    def forward(self, input: ptens.ptensors0) -> ptens.ptensors0:...

    def forward(self, input: Union[ptens.ptensors0,ptens.ptensors1,ptens.ptensors2]) -> Union[ptens.ptensors0,ptens.ptensors1,ptens.ptensors2]:
      if isinstance(input,ptens.ptensors0):
        return ptens.ptensors0.from_matrix(super().forward(input.torch()),input.get_atoms())
      elif isinstance(input,ptens.ptensors1):
        return ptens.ptensors1.from_matrix(super().forward(input.torch()),input.get_atoms())
      return ptens.ptensors2.from_matrix(super().forward(input.torch()),input.get_atoms())
class LazyBatchNorm(torch.nn.Module):
    def __init__(self, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None) -> None:
        super().__init__()
        self.bn = torch.nn.LazyBatchNorm1d(eps, momentum, affine, track_running_stats, device, dtype)
    @override
    def forward(self, input: ptens.ptensors2) -> ptens.ptensors2:...
    @override
    def forward(self, input: ptens.ptensors1) -> ptens.ptensors1:...
    @override
    def forward(self, input: ptens.ptensors0) -> ptens.ptensors0:...

    def forward(self, input: Union[ptens.ptensors0,ptens.ptensors1,ptens.ptensors2]) -> Union[ptens.ptensors0,ptens.ptensors1,ptens.ptensors2]:
      if isinstance(input,ptens.ptensors0):
        return ptens.ptensors0.from_matrix(self.bn(input.torch()),input.get_atoms())
      elif isinstance(input,ptens.ptensors1):
        return ptens.ptensors1.from_matrix(self.bn(input.torch()),input.get_atoms())
      return ptens.ptensors2.from_matrix(self.bn(input.torch()),input.get_atoms())
class PNormalize(torch.nn.Module):
  def __init__(self, p: int = 2, eps: float = 1E-5) -> None:
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
class Reduce_1P_0P(torch.nn.Module):
  def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
    super().__init__()
    self.lin = Linear(in_channels,out_channels,bias)
  def forward(self, features: ptensors1) -> ptensors0:
    features = linmaps0(features)
    a = self.lin(features)
    return a
  
class GraphAttentionLayer_P0(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, d_prob = 0.5, leakyrelu_alpha = 0.5, relu_alpha = 0.5, concat=True):
        super(GraphAttentionLayer_P0, self).__init__()
        self.d_prob = d_prob
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.leakyrelu_alpha = leakyrelu_alpha
        self.relu_alpha = relu_alpha
        self.concat = concat

        self.W = Parameter(torch.empty(in_channels, out_channels))
        nn.init.xavier_uniform_(self.W.data)
        self.a = Parameter(torch.empty(out_channels, 1))
        nn.init.xavier_uniform_(self.a.data)
        
        self.leakyrelu = LeakyReLU(self.leakyrelu_alpha)
        self.relu = relu(self.relu_alpha)

    def forward(self, h: ptensors0, adj: ptensors0):
        h_torch = h.torch()
        adj_torch = adj.torch()
        Wh = torch.mm(h_torch, self.W) 
        e = self._prepare_attentional_mechanism_input(Wh) 
        zero_vec = -9e15*torch.ones_like(e) 
        attention = dropout(softmax(torch.where(adj_torch > 0, e, zero_vec) , dim=1),
                            self.d_prob, training = self.training)
        h_prime = torch.matmul(attention, Wh)
        
        h_prime_p0 = ptens.ptensors0.from_matrix(h_prime)
        if self.concat:
            return relu(h_prime_p0)
        else:
            return h_prime_p0
      
    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_channels, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_channels:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_channels) + ' -> ' + str(self.out_channels) + ')'
        
class GraphAttentionLayer_P1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, d_prob: torch.float = 0.5, alpha = 0.5, cat=True):
        super(GraphAttentionLayer_P1, self).__init__()
        self.d_prob = d_prob
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.concat = cat

        self.W = Parameter(torch.empty(in_channels, out_channels))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = Parameter(torch.empty(2*out_channels, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.relu = relu(self.alpha)

    def forward(self, h: ptensors1, adj: ptensors1):
        h = ptens.linmaps0(h).torch() 
        Wh = torch.mm(h, self.W) 
        e_p1 = self._prepare_attentional_mechanism_input(Wh)                   
        e_p0 = ptens.linmaps0(e_p1)
        e_p0_r, e_p0_c = e_p0.torch().size()
        e_p1_r = e_p0_r + e_p1.get_nc()

        zero_vec = -9e15*torch.ones_like(e_p0_r, e_p0_c)
        attention = softmax(torch.where(ptens.linmaps0(adj).torch() > 0, e_p0.torch(), zero_vec), dim=1)
        attention_p1 = Dropout(ptens.linmaps1(ptens.ptensors0.from_matrix(attention)), self.d_prob)  
        h_prime = attention_p1*Wh
        if self.concat:
            return relu(h_prime)
        else:
            return h_prime 

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_channels, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_channels:, :])
        e = Wh1 + Wh2.T
        e_p1 = ptens.linmaps1(ptens.ptensors0.from_matrix(e))
        return self.relu(e_p1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_channels) + ' -> ' + str(self.out_channels) + ')'
