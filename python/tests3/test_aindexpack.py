import torch
import ptens_base as pb
import ptens as p

print("\n-------------------------------------")
print (" aindexpack")
print("-------------------------------------\n")

v={1:[1,2,3], 3:[5,2]}
A=pb.aindexpack.from_lists(v)
print(A)
