************
Unite layers
************

Unite layers are similar to gather layers, except that the reference domain of each output 
vertex becomes the union of the reference domains of its neighbors. 
Consequently, a unite layers are true higher order message passing layers that can 
connect Ptensor layers of any order. The following shows a ``unite1`` operation applied to a 
``ptensors`` layer:

.. code-block:: python

 >>> A=p.ptensors1.randn([[1,2],[2,3],[3]],3)
 >>> print(A)
 tensor1 [1,2]:
 [ -1.23974 -0.407472 1.61201 ]
 [ 0.399771 1.3828 0.0523187 ]

 Ptensor1 [2,3]:
 [ -0.904146 1.87065 -1.66043 ]
 [ -0.688081 0.0757219 1.47339 ]

 Ptensor1 [3]:
 [ 0.097221 -0.89237 -0.228782 ]

 >>> G=p.graph.from_matrix(torch.tensor([[1.0,1.0,0],[1.0,0,0],[0,0,1.0]]))
 >>> B=p.unite1(A,G)
 >>> print(B)
 Ptensor1 [1,2,3]:
 [ -0.839965 0.97533 1.66433 -1.23974 -0.407472 1.61201 ]
 [ -2.43219 2.92171 1.47729 -0.504376 3.25345 -1.60811 ]
 [ -1.59223 1.94638 -0.187041 -0.688081 0.0757219 1.47339 ]

 Ptensor1 [1,2]:
 [ -0.839965 0.97533 1.66433 -1.23974 -0.407472 1.61201 ]
 [ -0.839965 0.97533 1.66433 0.399771 1.3828 0.0523187 ]

 Ptensor1 [3]:
 [ 0.097221 -0.89237 -0.228782 0.097221 -0.89237 -0.228782 ]
