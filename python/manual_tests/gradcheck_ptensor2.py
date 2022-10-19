def gradcheck_linmaps0(seq):
    atoms, nc = seq.atoms, seq.get_nc()
    x = torch.tensor(seq, requires_grad=True, dtype=torch.double)
    c = torch.autograd.gradcheck(Ptensor2_Linmaps0Fn.apply, x)
    return c

def gradcheck_linmaps1(seq):
    atoms, nc = seq.atoms, seq.get_nc()
    x = torch.tensor(seq, requires_grad=True, dtype=torch.double)
    c = torch.autograd.gradcheck(Ptensor2_Linmaps1Fn.apply, x)
    return c

def gradcheck_linmaps2(seq):
    atoms, nc = seq.atoms, seq.get_nc()
    x = torch.tensor(seq, requires_grad=True, dtype=torch.double)
    c = torch.autograd.gradcheck(Ptensor2_Linmaps2Fn.apply, x)
    return c


def gradcheck_transfer0(seq, _atoms):
    pre_atoms, pre_nc = seq.atoms, seq.get_nc()
    x_seq = torch.tensor(seq, requires_grad=True, dtype=torch.double)
    x_atoms = torch.tensor(_atoms, requires_grad=True, dtype=torch.double) 
    x = x_seq, x_atoms
    c = torch.autograd.gradcheck(Ptensor2_Transfer0Fn.apply, x)    
    return c

def gradcheck_transfer1(seq, _atoms):
    pre_atoms, pre_nc = seq.atoms, seq.get_nc()
    x_seq = torch.tensor(seq, requires_grad=True, dtype=torch.double)
    x_atoms = torch.tensor(_atoms, requires_grad=True, dtype=torch.double) 
    x = x_seq, x_atoms
    c = torch.autograd.gradcheck(Ptensor2_Transfer1Fn.apply, x)   
    return c

def gradcheck_transfer2(seq, _atoms):
    pre_atoms, pre_nc = seq.atoms, seq.get_nc()
    x_seq = torch.tensor(seq, requires_grad=True, dtype=torch.double)
    x_atoms = torch.tensor(_atoms, requires_grad=True, dtype=torch.double) 
    x = x_seq, x_atoms
    c = torch.autograd.gradcheck(Ptensor2_Transfer2Fn.apply, x)    
    return c
