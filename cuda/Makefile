WTH_CUDA=t
CUDA_HOME=/usr/local/cuda

ROOTDIR=..
include $(ROOTDIR)/common.txt

INCLUDE=  $(CNINE_INCLUDES) $(PTENS_INCLUDES) #-I$(INCLUDEDIR) -I$(TENSORSDIR) -I$(LAYERSDIR) 

TESTS=

DEPS= 

EXECS= 

OBJECTS=
#ifdef EIGENDIR
#OBJECTS+=EigenRoutines.o 
#endif 

CUDA_OBJECTS=$(patsubst %.cu,%.o,$(wildcard *.cu))

CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
CUDA_OBJECTS=$(patsubst %.cu,%,$(wildcard *.cu))
NVCCFLAGS+= -D_WITH_CUDA -D_WITH_CUBLAS
endif 


$(CUDA_OBJECTS): %.o: 
	$(NVCC) $(NVCCFLAGS) -c $*.cu $(CFLAGS) $(MACROS) $(INCLUDE) 


objects: $(OBJECTS) $(CUDA_OBJECTS)

all: objects 

clean: 
	rm -f $(OBJECTS) $(CUDA_OBJECTS) 

anew: clean all
