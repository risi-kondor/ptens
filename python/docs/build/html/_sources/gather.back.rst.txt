*************
Gather layers
*************

The simplest type of equivariant message passing is the type described in (Gilmer et al. 2017), 
mapping a a zeroth order Ptensor layer to a zeroth order Ptensor layer simply by summing 
the messages coming from each neighbor node. 
This operation is implemented in the ``gather`` function:

.. code-block:: python

 >>> A=ptens.ptensors0.randn(5,3)
 >>> A
 Ptensor0 [0]:
 [ -1.23974 -0.407472 1.61201 ]

 Ptensor0 [1]:
 [ 0.399771 1.3828 0.0523187 ]

 Ptensor0 [2]:
 [ -0.904146 1.87065 -1.66043 ]

 Ptensor0 [3]:
 [ -0.688081 0.0757219 1.47339 ]

 Ptensor0 [4]:
 [ 0.097221 -0.89237 -0.228782 ]

 >>> G=ptens.graph.random(5,0.5)
 >>> G
 0<-((2,1)(3,1)(4,1))
 2<-((0,1)(3,1)(4,1))
 3<-((0,1)(2,1)(4,1))
 4<-((0,1)(2,1)(3,1))

 >>> B=ptens.gather(A,G)
 >>> B
 Ptensor0 [0]:
 [ -1.49501 1.05401 -0.415822 ]

 Ptensor0 [1]:
 [ 0 0 0 ]

 Ptensor0 [2]:
 [ -1.8306 -1.22412 2.85662 ]

 Ptensor0 [3]:
 [ -2.04666 0.570812 -0.277201 ]

 Ptensor0 [4]:
 [ -2.83196 1.5389 1.42497 ]



