import torch
import ptens_base as pb
import ptens as p


print("\n----------------------------------")
print(" layer_map")
print("----------------------------------\n")


print("A layer_map is a mapping between two tensor layers indicating")
print("which tensors in the first layer send messages to which tensors")
print("in the second.")

A=pb.atomspack.random(6,6,0.5)
B=pb.atomspack.random(6,6,0.5)

M=pb.layer_map.overlaps_map(A,B)
print(M)
