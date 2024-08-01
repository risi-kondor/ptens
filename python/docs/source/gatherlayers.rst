**************************************
Message passing between Ptensor layers
**************************************

The gather maps of the previous section make it possible to extend higher order equivariant message passing 
to Ptensor layers. First, however, we need a way to define which tensor in the input layer communicates  
with which tensor in the output layer. 

==========
Layer Maps
==========

The objects that `ptens` uses to define which Ptensors of the output layer each Ptensor in the input layer 
sends messages to are ``ptens_base.layer_map``\s. It is easiest to define ``layer_map``\s directly from 
the ``atomspack``\s of the two layers. One of the most common ways to do this is via the 
``overlaps_map`` constructor that is used to send messages between all pairs of Ptensors 
whose reference domains overlap in at least one "atom":

.. code-block:: python

 >> atoms1=ptens_base.atomspack([[1,2,3],[3,5],[2]])
 >> atoms2=ptens_base.atomspack([[3,2],[1,4],[3]])

 >> L=ptens_base.layer_map.overlaps_map(atoms2,atoms1)
 >> print(L)

 0<-(0,1,2)
 1<-(0)
 2<-(0,1)

In this example, the first output reference domain (``[3,2]``) overlaps with each of the 
input reference domains, therefore ``L`` maps inpput Ptensors (0,1,2) to output Ptensor 0.  
The second output reference domain (``[1,4]``) only overlaps with the first input, therefore 
``L`` will send ``1<-0``, and so on. 

By default, ``layer_map`` objects are cached for as long as the ``atomspack`` objects from which 
they were computed are in scope.


====================================
Gather maps between Ptensor layers
====================================

`ptens` uses the same ``gather`` operations as described in the 
`previous section <gather.html#gather-maps-between-ptensors>`_ to send messages from one Ptensor 
layer to another. To instantiate this we must specify:

#. The input Ptensor layer
#. The reference domains of the Ptensors in the output layer 
#. The ``layer_map`` connecting the input layer and the output layer. 

The following illustrates how to send messages from a first order layer to another first order layer:

.. code-block:: python

 >> in_atoms=ptens_base.atomspack.from_list([[1,3,4],[2,5],[0,2]])
 >> out_atoms=ptens_base.atomspack.from_list([[2,4],[3,5],[1]])
 >> L=ptens_base.layer_map.overlaps_map(out_atoms,in_atoms)
 >> A=ptens.ptensorlayer1.randn(in_atoms,3)
 >> print(A)

 Ptensorlayer1:
   Ptensor1 [1,3,4]:
     [ 0.989148 1.30568 0.0376512 ]
     [ -1.18443 2.25047 1.26969 ]
     [ -0.148695 -0.504967 -1.62654 ]
   Ptensor1 [2,5]:
     [ 0.770672 -0.782321 -0.569275 ]
     [ -0.555409 1.29336 0.181371 ]
   Ptensor1 [0,2]:
     [ 0.568828 1.0944 2.59344 ]
     [ 0.604974 -0.00491901 -0.082703 ]

 >> B=ptens.ptensorlayer1.gather(out_atoms,A,L)
 
 Ptensorlayer1:
   Ptensor1 [2,4]:
     [ 1.37565 -0.78724 -0.651978 1.37565 -0.78724 -0.651978 ]
     [ -0.148695 -0.504967 -1.62654 -0.148695 -0.504967 -1.62654 ]
   Ptensor1 [3,5]:
     [ -1.18443 2.25047 1.26969 -1.18443 2.25047 1.26969 ]
     [ -0.555409 1.29336 0.181371 -0.555409 1.29336 0.181371 ]
   Ptensor1 [1]:
     [ 0.989148 1.30568 0.0376512 0.989148 1.30568 0.0376512 ]

The ``layer_map`` is an optional argument in this case. If no ``layer_map`` is specified, by default  
``gather`` will use the ``overlaps_map`` between the reference domains of the input and output layers, 
simplifying the above code to just:

.. code-block:: python

 >> A=ptens.ptensorlayer1.randn([[1,3,4],[2,5],[0,2]],3)
 >> B=ptens.ptensorlayer1.gather([[2,4],[3,5],[1]],A)


