import torch
import ptens_base as pb
import ptens as p


print("\n----------------------------------")
print(" vram_manager")
print("----------------------------------\n")

print("Some message passing operations require allocating temporary objects.")
print("When using the GPU, these memory allocations can sometimes be significantly")
print("accelerated by using cnine's own memory manager.\n")

print("We can activate or reset the memory manager specifying the size of GPU")
print("memory dedicated to it in Mbytes:\n")
pb.vram_manager.reset(1000)

print("\nWe can find out the size of managed memory:\n")
print(pb.vram_manager.size())


