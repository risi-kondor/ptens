from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import Parameter, Sequential, LeakyReLU, ReLU
from torch.nn.functional import softmax, dropout
import ptens 
from ptens import ptensors0, ptensors1, ptensors2, graph
from warnings import warn
from ptens_base import atomspack
from ptens.functions import linear, linmaps0, outer, unite1, gather, relu, cat
######################################## Functions ###########################################
def ConvertEdgeAttributesToPtensors1(edge_index: torch.Tensor, edge_attributes: torch.Tensor):
  atoms = edge_index.transpose(0,1).float()
  return ptensors1.from_matrix(edge_attributes,atoms)

def ComputeSubstructureMap(source_domains: atomspack, graph_filter: graph, G: graph) -> Tuple[graph,graph]:
  # TODO: make it so .get_atoms() returns an atomspack, so we don't have to convert back to one.
  return graph.overlaps(G.subgraphs(graph_filter),atomspack(source_domains))

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
    if not isinstance(self.w,torch.nn.parameter.UninitializedParameter):
      self.w = torch.nn.init.xavier_uniform_(self.w)
      if not self.b is None:
        self.b = torch.nn.init.zeros_(self.b)
  def initialize_parameters(self, input) -> None:
    if isinstance(self.w,torch.nn.parameter.UninitializedParameter):
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
class LazyUnite(torch.nn.Module):
  def __init__(self, channels_out: int = None, out_order: int = None, bias : bool = True, reduction_type : str = "sum") -> None:
    r"""
    reduction_types: "sum" and "mean"
    leave 'out_order' and 'channels_out' as 'None' to keep same as input
    NOTE: if you are planning on using the same source/target domains more than once, consider using a unite layer instead and computing the domain mapping separately.
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
  def forward(self, features: Union[ptensors0,ptensors1,ptensors2], graph: graph) -> Union[ptensors0,ptensors1,ptensors2]:
    domain_map = graph.overlaps(graph.subgraphs(self.graph_filter),atomspack(features.get_atoms()))
    if self.unite is None:
      if self.lin.out_channels is None:
        self.lin.out_channels = self.features.get_nc()
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
        [lambda x,y: x,ptensors0.unite1,ptensors0.unite2],
        [ptensors1.linmaps0,ptensors1.unite1,ptensors1.unite2],
        [ptensors2.linmaps0,ptensors2.unite1,ptensors2.unite2],
        ][in_order][out_order]
      if in_order == 0 and out_order == 0:
        warn("You have 'in_order = out_order = 0', this means the unite layer is simply a linear layer.  Did you intend this?")
      if out_order == 0:
        q = self.unite
        self.unite = lambda x,G,n: q(x,n)
    F = self.unite(features,domain_map,self.use_mean)
    F = self.lin(F)
    return F
class LazySubstructureTransport(LazyUnite):
  def __init__(self, channels_out: int, graph_filter: Union[graph,Tuple[str,int],None] = None, out_order: int = None, bias : bool = True, reduction_type : str = "sum") -> None:
    r"""
    reduction_types: "sum" and "mean"
    leave 'out_order' as 'None' to keep same as input (NOTE: cannot leave as default if input is of order 0.)
    NOTE: if you are planning on using the same source/target domains more than once, consider using a unite layer instead and computing the domain mapping separately.
    """
    super().__init__(channels_out,out_order,bias,reduction_type)
    assert out_order != 0
    if isinstance(graph_filter,Tuple[str,int]):
      graph_filter = generate_generic_shape(*graph_filter)
    assert isinstance(graph_filter,graph)
    self.graph_filter = graph_filter
  def forward(self, features: Union[ptensors0,ptensors1,ptensors2], graph: graph) -> Union[ptensors0,ptensors1,ptensors2]:
    domain_map = ComputeSubstructureMap(features.get_atoms(),self.graph_filter,graph)
    return super().forward(features,domain_map)
class LazyTransfer(torch.nn.Module):
  def __init__(self, channels_out: int = None, reduction_type : str = "sum", out_order: int = None, bias : bool = True) -> None:
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
        raise "'features' must be instance of 'ptensors[0|1|2]'"
      out_order = in_order if self.out_order is None else self.out_order
      #
      self.lin.out_channels = features.get_nc()
      #
      self.transfer = [
        [ptensors0.transfer0,ptensors0.transfer1,ptensors0.transfer2],
        [ptensors1.transfer0,ptensors1.transfer1,ptensors1.transfer2],
        [ptensors2.transfer0,ptensors2.transfer1,ptensors2.transfer2],
        ][in_order][out_order]
      if in_order == 0:
        q = self.transfer
        self.transfer = lambda x,G,n: q(x,n)
    F = self.transfer(features,graph,self.use_mean)
    F = self.lin(F)
    return F
class LazyTransferNHoods(LazyTransfer):
  def __init__(self, channels_out: int, num_hops: int, reduction_type : str = "sum", out_order: int = None, bias : bool = True) -> None:
    r"""
    reduction_types: "sum" and "mean"
    leave 'out_order' as 'None' to keep same as input
    """
    super().__init__(channels_out,reduction_type,out_order,bias)
    self.num_hops = num_hops
    self.lin2 = LazyLinear(channels_out,bias=False)
  def forward(self, features: Union[ptensors0,ptensors1,ptensors2], graph: graph) -> Union[ptensors0,ptensors1,ptensors2]:
    return super().forward(features,graph.nhoods(self.num_hops),graph)
class LazyPtensGraphConvolutional(LazyTransfer):
  def __init__(self, channels_out: int = None, reduction_type : str = "sum", out_order: int = None, bias : bool = True) -> None:
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
  def __init__(self, prob: torch.float = 0.5, device : str = 'cuda') -> None:
    super().__init__()
    self.p = prob
    self.device = device
  def cuda(self, device = None):
    self.device = device
    return super().cuda(device)
  def forward(self, x):
    if self.training:
      dropout = 1/(1 - self.p)*(torch.rand(x.get_nc(),device=self.device) > self.p)
      if isinstance(x,ptensors0):
        return ptensors0.mult_channels(x,dropout)
      elif isinstance(x,ptensors1):
        return ptensors1.mult_channels(x,dropout)
      elif isinstance(x,ptensors2):
        return ptensors2.mult_channels(x,dropout)
      else:
        raise NotImplementedError('Dropout not implemented for type \"' + str(type(x)) + "\"")
    else:
      return x
class LazyBatchNorm(torch.nn.Module):
  def __init__(self, eps: torch.float = 1E-5, momentum: torch.float = 0.1) -> None:
    super().__init__()
    #self.running_mean = torch.nn.parameter.UninitializedParameter()
    #self.running_var = torch.nn.parameter.UninitializedParameter()
    #self.weight = torch.nn.parameter.UninitializedParameter()
    #self.bias = torch.nn.parameter.UninitializedParameter()
    self.eps = eps
    self.momentum = momentum
  def forward(self, x):
    r"""
    x can be any type of ptensors
    """
    x_val : torch.Tensor = x.torch()
    if len(list(self.parameters())) == 0:
      nc = x.get_nc()
      #self.running_mean.materialize(nc)
      #self.running_var.materialize(nc)
      #self.weight.materialize(nc)
      #self.bias.materialize(nc)
      running_mean = x_val.mean(0)
      running_var = x_val.var(0)
      self.running_mean = torch.nn.parameter.Parameter(running_mean)
      self.running_var = torch.nn.parameter.Parameter(running_var)
      self.weight = torch.nn.parameter.Parameter(torch.ones(nc))
      self.bias = torch.nn.parameter.Parameter(torch.zeros(nc))
    else:
      m = self.momentum
      running_mean = (1 - m) * self.running_mean + m * x_val.mean(0)
      running_var = (1 - m) * self.running_var + m * x_val.var(0)
    #
    m = self.weight * (running_var + self.eps)**-0.5
    # TODO: if we can add channel wise addition broadcasting to all reference domains, we will not need to do this.
    b = self.bias - running_mean * m
    output = linear(x,torch.diag(m),b)
    return output
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
class Reduce_1P_0P(torch.nn.Module):
  def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
    super().__init__()
    self.lin = Linear(in_channels,out_channels,bias)
  def forward(self, features: ptensors1) -> ptensors0:
    features = linmaps0(features)
    a = self.lin(features)
    return a
class GraphAttentionLayer_P0(nn.Module):
    """
    An implementation of GATConv layer in ptens. 
    """
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

    # ptensors0 -> tensor -> do -> ptensors0
    def forward(self, h: ptensors0, adj: ptensors0):
        h_torch = h.torch()
        adj_torch = adj.torch()
        Wh = torch.mm(h_torch, self.W) 
        e = self._prepare_attentional_mechanism_input(Wh) 

        zero_vec = -9e15*torch.ones_like(e) 
        attention = torch.where(adj_torch > 0, e, zero_vec) 
        attention = softmax(attention, dim=1)
        attention = dropout(attention, self.d_prob, training = self.training)
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
    """
    An implementation of GATConv layer in ptens. 
    """
#   linmaps0->1: copy/ptensor by len(domain)
#   linmaps1->0: sum/ptensor
    def __init__(self, in_channels: int, out_channels: int, d_prob: torch.float = 0.5, alpha = 0.5, concat=True):
        super(GraphAttentionLayer_P1, self).__init__()
        self.d_prob = d_prob
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.concat = concat

        self.W = Parameter(torch.empty(in_channels, out_channels))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = Parameter(torch.empty(2*out_channels, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.relu = relu(self.alpha)

    def forward(self, h: ptensors1, adj: ptensors1):
        h = ptens.linmaps0(h).torch() # ptensors1 -> ptensors0 -> torch
        Wh = torch.mm(h, self.W) 
        e_p1 = self._prepare_attentional_mechanism_input(Wh)
        # e_p1 -> e_p0 -> size of e_p0 -> size of e_p1                      
        e_p0 = ptens.linmaps0(e_p1) 
        e_p0_r, e_p0_c = e_p0.torch().size()
        e_p1_r = e_p0_r + e_p1.get_nc()                                   
        e_p1_c = e_p0_c
        zero_vec = -9e15*torch.ones_like(e_p0_r, e_p0_c)
        # ptensors1 -> ptensors0 -> torch -> do -> ptensors0 -> ptensors1 
        adj_torch = ptens.linmaps0(adj).torch()
        e_torch = e_p0.torch()
        attention = torch.where(adj_torch > 0, e_torch, zero_vec)
        attention = softmax(attention, dim=1)
        attention_p1 = ptens.linmaps1(ptens.ptensors0.from_matrix(attention))
        attention_p1 = Dropout(attention_p1, self.d_prob)  
        h_prime = attention*Wh
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
