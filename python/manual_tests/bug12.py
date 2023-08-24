import ptens
from ptens_base import atomspack
atoms_1 = atomspack([[0]])
atoms_2 = atomspack([[1]])
features_1 = ptens.ptensors1.randn(atoms_1,32)
G = ptens.graph.overlaps(atoms_2,atoms_1)
features_1.transfer1(atoms_2,G)
