import torch
import ptens_base as pb
import ptens as p


print("\n----------------------------------")
print(" batched_atomspack")
print("----------------------------------\n")

print("A batched_atomspack is a batch of atomspack objects.\n")

print("We can defined from a list of list of lists:\n")
A=pb.batched_atomspack([[[0,7]],[[1,3,4],[2,5]]])
print(A)

print("\nOr a collection of regular atomspacks:\n")
a=pb.atomspack.random(3,6,0.5)
A=pb.batched_atomspack([a,a,a])
print(A)

print("\nWe can retrieve all the reference domains:\n") 
print(A.torch())

print("\nOr just select ones:\n")
print(A[1])
print("\n")


