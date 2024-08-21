import torch
import ptens

class SchurLayer(torch.nn.Module):

  def __init__(self, in_channels: int, out_channels: int, S:ptens.subgraph) -> None:
      super().__init__() #This follows Glorot initialization for weights.
      S.set_evecs()
      k=S.n_espaces()
      self.w = torch.nn.parameter.Parameter(torch.empty(k,in_channels,out_channels),requires_grad=True)
      self.b = torch.nn.parameter.Parameter(torch.empty(k,out_channels),requires_grad=True)
      self.reset_parameters()

  def reset_parameters(self):
      if not isinstance(self.w,torch.nn.parameter.UninitializedParameter):
          self.w = torch.nn.init.xavier_uniform_(self.w)
          self.b = torch.nn.init.zeros_(self.b)

  def forward(self, x: ptens.subgraphlayer1) -> ptens.subgraphlayer1:
      return x.schur_layer(self.w,self.b)

