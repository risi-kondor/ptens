import torch
import ptens as p

# generating atoms for random ptensors1.
N = 5
atoms = []
entry_count = 0
for i in range(N):
  ref_domain_len = torch.randint(N,(1,)).tolist()[0]
  ref_domain = torch.randint(N,(ref_domain_len,)).tolist()
  atoms.append(ref_domain)
  entry_count += ref_domain_len

print(atoms)

# computing entries for random ptensors1.
x = torch.randn(entry_count,3)

# computing ptensors1.
x = p.ptensors1.from_matrix(x,atoms)

# creating graph.
G = p.graph.random(N,0.5)

# moving features ptensors1 to cuda.
# x.to("cuda")

# attempting a call to unite1.
y=p.unite1(x,G)

print(y)
