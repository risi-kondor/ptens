********************************
Message passing between Ptensors
********************************

Hands et al. [] extended the notion of message passing between permutation equivariant tensors 
(Ptensors) to the case when the reference domains of the source and destination tensors 
are not necessarily the same. In this case the number of possible linearly independent 
linear maps increases because for each summation or broadcast operation we can consider summing/broadcasting 
over all elements of reference domain or only the intersection of the reference domains of the sending 
and receiving P-tensors. 
These linear maps in `ptens` are called ``gather`` operations.  

==============================
Gather maps between Ptensors
==============================

---------
gather0
---------

Similarly to ``linmaps0``, the ``gather0`` function passes equivariant linear messages to a zeroth order Ptensor. 
In contrast to ``linmaps0``, however, the reference domain of the output must be specified explicitly. 

.. 
  In the case of :math:`\mathcal{P}_0\to\mathcal{P}_0` message passing, the only possible linear map is 
  the identity:

The only possible equivariant map from a zeroth order Ptensor to a zeroth order Ptensor 
is a multiple of the identity:

.. code-block:: python

 >> A=ptens.ptensor0.sequential([1,2,3],5)
 >> print(A)

 Ptensor0(1,2,3):
 [ 0 1 2 3 4 ]

 >> B=ptens.linmaps0(A)
 >> print(B)

 Ptensor0(1,2,3):
 [ 0 1 2 3 4 ]

A first order Ptensor can gather information to a zeroth order Ptensor by extracting its slice 
corresponding to the reference domain of the latter:

.. code-block:: python

 >> A=ptens.ptensor1.sequential([1,2,3],3)
 >> A

 Ptensor1 [1,2,3]:
   [ 0 1 2 ]
   [ 3 4 5 ]
   [ 6 7 8 ]

 >> B=ptens.gather0(A,[2])
 >> B

 Ptensor0 [2]:
   [ 3 4 5 ]

A second order Ptensor can gather information to a zeroth order Ptensor either by 
summing the entire block corresponding to the intersection of their reference domains, 
or just its diagonal:

.. code-block:: python

 >> A=ptens.ptensor2.sequential([1,2,3],3)
 >> A

 Ptensor2(1,2,3):
 channel 0:
   [ 0 3 6 ]
   [ 9 12 15 ]
   [ 18 21 24 ]

 channel 1:
   [ 1 4 7 ]
   [ 10 13 16 ]
   [ 19 22 25 ] 
 
 channel 2:
   [ 2 5 8 ]
   [ 11 14 17 ]
   [ 20 23 26 ]

 >> B=ptens.gather0(A,[2])
 >> B

 Ptensor0(2):
 [ 12 13 14 12 13 14 ]


---------
gather1
---------

When a message is gatherred from a zeroth order Ptensor to a first order Ptensor, effectively 
it is just copied into the row corresponding to the intersection of the reference domains:

.. code-block:: python

 >> A=ptens.ptensor0.sequential([2],3)
 >> A

 Ptensor0(2):
 [ 0 1 2 ]
 
 >> B=ptens.gather1(A,[2,3])
 >> B

 Ptensor1(2,3):
 [ 0 1 2 ]
 [ 0 0 0 ]

A message from a first order Ptensor to a first order Ptensor consists of the concatenation 
of two maps: copying to the intersection and broadcasting the sum over the elements of the 
intersection:

.. code-block:: python

 >> A=ptens.ptensor1.sequential([1,2,3],3)
 >> A

 Ptensor1(1,2,3):
 [ 0 1 2 ]
 [ 3 4 5 ]
 [ 6 7 8 ]
 
 >> B=ptens.gather1(A,[2,3,5])
 >> B

 Ptensor1 [2,3,5]:
 [ 9 11 13 3 4 5 ]
 [ 9 11 13 6 7 8 ]
 [ 0 0 0 0 0 0 ]


When a message is passed from a second order Ptensor to a first order Ptensor we have 5 possible 
linear maps, hence the number of channels is multiplied by five. 

.. code-block:: python

 >> A=ptens.ptensor2.sequential([1,2,3],3)
 >> A

 Ptensor2 [1,2,3]:
 channel 0:
   [ 0 3 6 ]
   [ 9 12 15 ]
   [ 18 21 24 ]

 channel 1:
   [ 1 4 7 ]
   [ 10 13 16 ]
   [ 19 22 25 ]

 channel 2:
   [ 2 5 8 ]
   [ 11 14 17 ]
   [ 20 23 26 ]

 >> B=ptens.gather1(A,[2,3,5])
 >> B

 Ptensor1 [2,3,5]:
 [ 72 76 80 36 38 40 33 35 37 27 29 31 12 13 14 ]
 [ 72 76 80 36 38 40 39 41 43 45 47 49 24 25 26 ]
 [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]


---------
gather2
---------

Similarly to linmaps, the number of possible gathers maps from zeroth, first and second order 
Ptensors to second order Ptensors is 2,5 and 15, respectively:

.. code-block:: python

 >> A=ptens.ptensor0.sequential([2],3)
 >> A

 Ptensor0 [2]:
  [ 0 1 2 ]
 >> B=ptens.gather2(A,[2,3,5])
 >> B

 Ptensor2 [2,3,5]:
 channel 0:
   [ 0 0 0 ]
   [ 0 0 0 ]
   [ 0 0 0 ]

 channel 1:
   [ 1 0 0 ]
   [ 0 0 0 ]
   [ 0 0 0 ]

 channel 2:
   [ 2 0 0 ]
   [ 0 0 0 ]
   [ 0 0 0 ]

 channel 3:
   [ 0 0 0 ]
   [ 0 0 0 ]
   [ 0 0 0 ]

 channel 4:
   [ 1 0 0 ]
   [ 0 0 0 ]
   [ 0 0 0 ]

 channel 5:
   [ 2 0 0 ]
   [ 0 0 0 ]
   [ 0 0 0 ]


.. code-block:: python

 >> A=ptens.ptensor1.sequential([1,2,3],3)
 >> A

 Ptensor1 [1,2,3]:
 [ 0 1 2 ]
 [ 3 4 5 ]
 [ 6 7 8 ]

 >> B=ptens.gather2(A,[2,3,5])
 >> B

 Ptensor2 [2,3,5]:
 channel 0:
   [ 9 9 0 ]
   [ 9 9 0 ]
   [ 0 0 0 ]

 channel 1:
   [ 11 11 0 ]
   [ 11 11 0 ]
   [ 0 0 0 ]

 channel 2:
   [ 13 13 0 ]
   [ 13 13 0 ]
   [ 0 0 0 ]

 channel 3:
   [ 9 0 0 ]
   [ 0 9 0 ]
   [ 0 0 0 ]

 channel 4:
   [ 11 0 0 ]
   [ 0 11 0 ]
   [ 0 0 0 ]

 channel 5:
   [ 13 0 0 ]
   [ 0 13 0 ]
   [ 0 0 0 ]

 channel 6:
   [ 3 6 0 ]
   [ 3 6 0 ]
   [ 0 0 0 ]

 channel 7:
   [ 4 7 0 ]
   [ 4 7 0 ]
   [ 0 0 0 ]
 ...


.. code-block:: python

 >> A=ptens.ptensor2.sequential([1,2,3],3)
 >> A

 Ptensor2 [1,2,3]:
 channel 0:
   [ 0 3 6 ]
   [ 9 12 15 ]
   [ 18 21 24 ]

 channel 1:
   [ 1 4 7 ]
   [ 10 13 16 ]
   [ 19 22 25 ]

 channel 2:
   [ 2 5 8 ]
   [ 11 14 17 ]
   [ 20 23 26 ]

 >> B=ptens.gather2(A,[2,3,5])
 >> B

 Ptensor2 [2,3,5]:
 channel 0:
   [ 72 72 0 ]
   [ 72 72 0 ]
   [ 0 0 0 ]

 channel 1:
   [ 76 76 0 ]
   [ 76 76 0 ]
   [ 0 0 0 ]

 channel 2:
   [ 80 80 0 ]
   [ 80 80 0 ]
   [ 0 0 0 ]

 channel 3:
   [ 36 36 0 ]
   [ 36 36 0 ]
   [ 0 0 0 ]

 channel 4:
   [ 38 38 0 ]
   [ 38 38 0 ]
   [ 0 0 0 ]

 channel 5:
   [ 40 40 0 ]
   [ 40 40 0 ]
   [ 0 0 0 ]

 channel 6:
   [ 72 0 0 ]
   [ 0 72 0 ]
   [ 0 0 0 ]

 channel 7:
   [ 76 0 0 ]
   [ 0 76 0 ]
   [ 0 0 0 ]


