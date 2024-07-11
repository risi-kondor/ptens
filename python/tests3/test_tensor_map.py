import torch
import ptens_base as pb
import ptens as p


print("\n----------------------------------")
print(" tensor_map")
print("----------------------------------\n")


print("A tensor_map is a maping from the indices of the P-tensors in one")
print("P-tensor layer to the indices of the P-tensors in another layer.\n")

A=pb.atomspack.random(6,6,0.5)
B=pb.atomspack.random(6,6,0.5)
pb.overlaps_tmap_cache.enable(True)

M=pb.tensor_map.overlaps_map(A,B)
print(M)

print(M.atoms())
print(M.in_indices())
print(M.out_indices())


print(pb.overlaps_tmap_cache.size())
print(pb.status_str())
