#----------------------------------------------------------------------------------
'''
Two ways: 
 - 1 check the grad before initializing ptensor
 - 2 check the grad after initializing ptensor      <-
'''

#def gradcheck_linmaps0(d, n):
#    seq = ptens.ptensor0.sequential(d,n)
#    d, n = seq.atoms, seq.get_nc()
#    x = torch.tensor(seq, requires_grad=True, dtype=torch.double)
#    c = torch.autograd.gradcheck(Ptensor0_Linmaps0Fn.apply, x)
#    return c

def gradcheck_linmaps0(seq):
    atoms, nc = seq.atoms, seq.get_nc()
    x = torch.tensor(seq, requires_grad=True, dtype=torch.double)
    c = torch.autograd.gradcheck(Ptensor0_Linmaps0Fn.apply, x)
    return c

def gradcheck_linmaps1(seq):
    atoms, nc = seq.atoms, seq.get_nc()
    x = torch.tensor(seq, requires_grad=True, dtype=torch.double)
    c = torch.autograd.gradcheck(Ptensor0_Linmaps1Fn.apply, x)
    return c

def gradcheck_linmaps2(seq):
    atoms, nc = seq.atoms, seq.get_nc()
    x = torch.tensor(seq, requires_grad=True, dtype=torch.double)
    c = torch.autograd.gradcheck(Ptensor0_Linmaps2Fn.apply, x)
    return c

def gradcheck_transfer0(seq, _atoms):
    pre_atoms, pre_nc = seq.atoms, seq.get_nc()
    x_seq = torch.tensor(seq, requires_grad=True, dtype=torch.double)
    x_atoms = torch.tensor(_atoms, requires_grad=True, dtype=torch.double) 
    x = x_seq, x_atoms
    c = torch.autograd.gradcheck(Ptensor0_Transfer0Fn.apply, x)    
    return c

def gradcheck_transfer1(seq, _atoms):
    pre_atoms, pre_nc = seq.atoms, seq.get_nc()
    x_seq = torch.tensor(seq, requires_grad=True, dtype=torch.double)
    x_atoms = torch.tensor(_atoms, requires_grad=True, dtype=torch.double) 
    x = x_seq, x_atoms
    c = torch.autograd.gradcheck(Ptensor0_Transfer1Fn.apply, x)   
    return c

def gradcheck_transfer2(seq, _atoms):
    pre_atoms, pre_nc = seq.atoms, seq.get_nc()
    x_seq = torch.tensor(seq, requires_grad=True, dtype=torch.double)
    x_atoms = torch.tensor(_atoms, requires_grad=True, dtype=torch.double) 
    x = x_seq, x_atoms
    c = torch.autograd.gradcheck(Ptensor0_Transfer2Fn.apply, x)    
    return c

