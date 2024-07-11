import torch
import ptens_base as pb
import ptens as p


print("\n----------------------------------")
print(" atomspack")
print("----------------------------------\n")

print("An atomspack can be defined from a list of lists:\n")
A=pb.atomspack.from_list([[1,3,4],[2,5]])
print(A)

print("\nOr randomly:\n")
A=pb.atomspack.random(3,6,0.5)
print(A)

print("\nWe can retrieve all the reference domains:\n") 
print(A.torch())

print("\nOr just select ones:\n")
print(A[1])
print("\n")







