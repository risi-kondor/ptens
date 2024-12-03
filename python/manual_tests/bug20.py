import torch
import ptens
import ptens_base

G = ptens.ggraph.random(2, 1.0)
nc = 2
x1 = ptens.ptensorlayer2.sequential(G.subgraphs(ptens.subgraph.trivial()), nc)
x2 = ptens.ptensorlayer2.sequential(G.subgraphs(ptens.subgraph.edge()), nc)

x = ptens.batched_ptensorlayer2.from_ptensorlayers([x1, x2])
# x = x.to("cuda")
x.requires_grad_()

atoms_out = ptens_base.batched_atomspack([ G.subgraphs(ptens.subgraph.trivial()), G.subgraphs(ptens.subgraph.trivial())])

z = ptens.batched_ptensorlayer2.gather(atoms_out, x)

zero = z.zeros_like()

l1sum = torch.nn.L1Loss(reduction='sum')
def reduction_fn(z):
    return l1sum(zero, z)

reduction_fn = torch.sum

def fn(x):
    z = ptens.batched_ptensorlayer2.gather(atoms_out, x)
    s = reduction_fn(z)
    return s
s=fn(x)
s.backward(torch.tensor(1))
g=x.grad
print(g)


def numerical_grad_sum(fn, x, h):
    grad = torch.zeros_like(x)
    for i in range(x.numel()):
        xp = x.clone()
        xp.view(-1)[i] += h
        xm = x.clone()
        xm.view(-1)[i] -= h

        # Using torch.sum here, because torch autograd, calcualtes the partial diff of a scalar valued functino.
        # With sum, we can a scalar valued function, and the summed parts factorize
        a = fn(xp).double()
        b = fn(xm).double()
        num_diff = torch.sum(a)- torch.sum(b)
        grad_value = num_diff / (2 * float(h))
        grad.view(-1)[i] = grad_value
    return grad
num_grad = numerical_grad_sum(fn, x, 1e-3)
print(torch.Tensor(num_grad))
