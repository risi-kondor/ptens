import torch
import ptens

print("\nA pth order ptensor over domain (a1, ..., ak) with c channels is a (p+1)th order tensor A \in R^{kx...xkxc} \n")

print("\nDefining a zeroth order ptensor with the reference domain (1) and 5 channels:\n")
A=ptens.ptensor0.randn([1],5) #size 5
print("A =")
print(A)

print("\nDefining a zeroth order ptensor with the reference domain (1,2) and 5 channels:\n")
A=ptens.ptensor0.randn([1,2],5) #size 5
print("A =")
print(A)

print("\nDefining a zeroth order ptensor with the reference domain (1,2,3) and 5 channels:\n")
A=ptens.ptensor0.randn([1,2,3],5) #size 5
print("A =")
print(A)

print("\nDefining a first order ptensor with atom 2 and 5 channels: (The size of the first p dimensions <- the size of the reference domain.)\n")
B=ptens.ptensor1.randn([2],5) #size 1x5
print("B =")
print(B)

print("\nDefining a first order ptensor with two atoms 2,3 and 5 channels:\n")
B=ptens.ptensor1.randn([2,3],5) #size 2x5
print("B =")
print(B)

print("\nDefining a first order ptensor with three atoms 1,2,3 and 5 channels:\n")
B=ptens.ptensor1.randn([1,2,3],5) #size 3x5
print("B =")
print(B)

print("\nDefining a second order ptensor over the reference domain (3) and 5 channels:\n")
C=ptens.ptensor2.randn([3],5) #size 1x1x5
print("C =")
print(C)

print("\nDefining a second order ptensor over the reference domain (1,3) and 5 channels:\n")
C=ptens.ptensor2.randn([1,3],5) #size 2x2x5
print("C =")
print(C)

print("\nDefining a second order ptensor over the reference domain (1,2,3) and 5 channels:\n")
C=ptens.ptensor2.randn([1,2,3],5) #size 3x3x5
print("C =")
print(C)

print("===")
print("\nInitializer for each classes: \n") #  for debugging purposes

print("\nDefining an initializer for a zeroth order ptensor with atoms 1,2,3 and 5 channels: \n")
D0=ptens.ptensor0.sequential([1,2,3],5) #size 5
print("D0 =")
print(D0)
print("\nDefining an initializer for a first order ptensor with atoms 2,3 and 5 channels: \n")
D1=ptens.ptensor1.sequential([2,3],5) #size 2x5
print("D1 =")
print(D1)
print("\nDefining an initializer for a second order ptensor with atom 100 and 5 channels: \n")
D2=ptens.ptensor2.sequential([100],5) #size 1x1x5
print("D2 =")
print(D2)

print("===")
print("\nEquivariant Linear Maps: ")

print("\nThe only equivariant linear map from 0th order to 0th order ptensors: \n")
A=ptens.ptensor0.sequential([1,2,3],3) 
print("A =")
print(A) # size 3
print("=>")
B=ptens.linmaps0(A) 
print("B =")
print(B) # size 3*1

print("\nThe only euivariant linear map from 0th order to 1th order ptensors: \n")
print("A =")
print(A) # size 3
print("=>")
B=ptens.linmaps1(A) 
print("B =")
print(B) # size 3x(3*1)

print("\nThe space of euivariant maps from 0th order to 2th order ptensors are spanned by two different maps: \n")
print("A =")
print(A) # size 3
print("=>")
B=ptens.linmaps2(A) 
print("B =")
print(B) # size 3x3x(3*2)

print("---")

print("\nThe only euivariant linear map from 1th order to 0th order ptensors: \n")
A=ptens.ptensor1.sequential([1,2,3],3)
print("A =")
print(A) # size 3x3
print("=>")
B=ptens.linmaps0(A)
print("B =")
print(B) # size 3*1

print("\nThe space of euivariant maps from 1th order to 1th order ptensors are spanned by two different maps: \n")
print("A =")
print(A) # size 3x3
print("=>")
B=ptens.linmaps1(A)
print("B =")
print(B) # size 3x(3*2)

print("\nThe space of euivariant maps from 1th order to 2th order ptensors are spanned by five different maps: \n")
print("A =")
print(A) # size 3x3
print("=>")
B=ptens.linmaps2(A)
print("B =")
print(B) # size 3x3x(3*5)

print("---")

print("\nThe space of euivariant maps from 2th order to 0th order ptensors are spanned by two different maps: \n")
A=ptens.ptensor2.sequential([1,2,3],3)
print("A =")
print(A) # size 3x3x3
print("=>")
B=ptens.linmaps0(A)
print("B =")
print(B) # size 3*2

print("\nThe space of euivariant maps from 2th order to 1th order ptensors are spanned by five different maps: \n")
print("A =")
print(A) # size 3x3x3
print("=>")
B=ptens.linmaps1(A)
print("B =")
print(B) # size 3x(3*5)

print("\nThe space of euivariant maps from 2th order to 2th order ptensors are spanned by fifteen different maps: \n")
print("A =")
print(A) # size 3x3x3
print("=>")
B=ptens.linmaps2(A)
print("B =")
print(B) # size 3x3x(3*15)



print("===")
print("\nTransfer Operations: ")
A=ptens.ptensor0.sequential([1,2,3],5)
print("\nTransfering from zeroth order ptensor with domain (1,2,3) and 5 channels to zeroth order ptensor with domain (2,3):\n")
print("A =")
print(A) # size 5
B=ptens.transfer0(A,[2,3])
print("B0 =")
print(B) # size 5
print("\nTransfering from zeroth order ptensor with domain (1,2,3) and 5 channels to first order ptensor with domain (2,3):\n")
print("A =")
print(A) # size 5
B=ptens.transfer1(A,[2,3])
print("B1 =")
print(B) # size 2x5
print("\nTransfering from zeroth order ptensor with domain (1,2,3) and 5 channels to second order ptensor with domain (2,3):\n")
print("A =")
print(A) # size 5
B=ptens.transfer2(A,[2,3])
print("B2 =")
print(B) # size 2x2x(5*2)

print("---")
A=ptens.ptensor1.sequential([1,2,3],5)
print("\nTransfering from first order ptensor with domain (1,2,3) and 5 channels to zeroth order ptensor with domain (2,3):\n")
print("A =")
print(A) # size 3x5
B=ptens.transfer0(A,[2,3])
print("B0 =")
print(B) # size 5
print("\nTransfering from first order ptensor with domain (1,2,3) and 5 channels to first order ptensor with domain (2,3):\n")
print("A =")
print(A) # size 3x5
B=ptens.transfer1(A,[2,3])
print("B1 =")
print(B) # size 2x(5*2)
print("\nTransfering from first order ptensor with domain (1,2,3) and 5 channels to second order ptensor with domain (2,3):\n")
print("A =")
print(A) # size 3x5
B=ptens.transfer2(A,[2,3])
print("B2 =")
print(B) # size 2x2x(5*5)

print("---")

A=ptens.ptensor2.sequential([1,2,3],5)
print("\nTransfering from second order ptensor with domain (1,2,3) and 5 channels to zeroth order ptensor with domain (2,3):\n")
print("A =")
print(A) # size 3x3x5
B=ptens.transfer0(A,[2,3])
print("B0 =")
print(B) # size 5*2
print("\nTransfering from second order ptensor with domain (1,2,3) and 5 channels to first order ptensor with domain (2,3):\n")
print("A =")
print(A) # size 3x3x5
B=ptens.transfer1(A,[2,3])
print("B1 =")
print(B) # size 2x(5*5)
print("\nTransfering from second order ptensor with domain (1,2,3) and 5 channels to second order ptensor with domain (2,3):\n")
print("A =")
print(A) # size 3x3x5
B=ptens.transfer2(A,[2,3])
print("B2 =")
print(B) # size 2x2x(5*5*5)


