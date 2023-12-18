ROOTDIR=../..
include $(ROOTDIR)/common.txt

INCLUDE= $(CNINE_INCLUDES) $(PTENS_INCLUDES) 
#-I$(INCLUDEDIR) -I$(TENSORSDIR) -I$(LAYERSDIR)  -I$()

TESTS=$(patsubst %.cpp,%,$(wildcard *.cpp))

DEPS= ../*.hpp 

OBJECTS= 

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
CUDA_EXTERNS+=$(CNINE_INCLUDEDIR)/Cnine_base.cu 
CUDA_OBJECTS=
CUDA_OBJECTS+=$(CNINE_CUDADIR)/RtensorUtils.o
endif 

$(TESTS): %: %.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o $@ $@.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o $@ $@.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif


tests: $(TESTS)
execs: $(EXECS)

all: execs tests 

clean: 
	rm -f $(TESTS) $(EXECS) 

anew: clean all

