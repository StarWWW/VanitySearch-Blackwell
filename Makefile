#---------------------------------------------------------------------
# Makefile for vanitysearch
#
# Author : Jean-Luc PONS
#
# Modified in 2026 for NVIDIA Blackwell (sm_120) throughput, see OPTIMIZATION.md and the git history.
# Default target is sm_120 (Blackwell). Other GPUs: make CCAP=89 (Ada), CCAP=86 (Ampere), ...
# Custom CUDA location: make CUDA=/usr/local/cuda-12.8

SRC = Base58.cpp IntGroup.cpp main.cpp Random.cpp \
      Timer.cpp Int.cpp IntMod.cpp Point.cpp SECP256K1.cpp \
      Vanity.cpp GPU/GPUGenerate.cpp hash/ripemd160.cpp \
      hash/sha256.cpp hash/sha512.cpp hash/ripemd160_sse.cpp \
      hash/sha256_sse.cpp Bech32.cpp

OBJDIR = obj

OBJET = $(addprefix $(OBJDIR)/, \
        Base58.o IntGroup.o main.o Random.o Timer.o Int.o \
        IntMod.o Point.o SECP256K1.o Vanity.o GPU/GPUGenerate.o \
        hash/ripemd160.o hash/sha256.o hash/sha512.o \
        hash/ripemd160_sse.o hash/sha256_sse.o \
        GPU/GPUEngine.o Bech32.o)

# The rules below define their targets in this order, "make" alone must still build the program
.DEFAULT_GOAL := all

CXX        = g++
CUDA       = /usr/local/cuda
CXXCUDA    = /usr/bin/g++
NVCC       = $(CUDA)/bin/nvcc
CCAP       = 120
GENCODE    = -gencode=arch=compute_$(CCAP),code=sm_$(CCAP) -gencode=arch=compute_$(CCAP),code=compute_$(CCAP)

ifdef debug
CXXFLAGS   = -mssse3 -Wno-write-strings -g -fno-strict-aliasing -I. -I$(CUDA)/include -include cstdint
else
CXXFLAGS   = -mssse3 -Wno-write-strings -O2 -fno-strict-aliasing -I. -I$(CUDA)/include -include cstdint
endif
LFLAGS     = -lpthread -L$(CUDA)/lib64 -lcudart_static -ldl -lrt

# No generated dependency files: a header change rebuilds every object that may include it
HEADERS     = $(wildcard *.h) $(wildcard hash/*.h) GPU/GPUEngine.h
GPU_HEADERS = $(HEADERS) $(wildcard GPU/*.h)

#--------------------------------------------------------------------

all: VanitySearch

ifdef debug
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu $(GPU_HEADERS)
	$(NVCC) -G -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -g -I$(CUDA)/include $(GENCODE) -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
else
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu $(GPU_HEADERS)
	$(NVCC) -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 -I$(CUDA)/include $(GENCODE) -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
endif

$(OBJDIR)/%.o : %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

VanitySearch: $(OBJET)
	@echo Making VanitySearch...
	$(CXX) $(OBJET) $(LFLAGS) -o vanitysearch

$(OBJET): | $(OBJDIR) $(OBJDIR)/GPU $(OBJDIR)/hash

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/GPU: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p GPU

$(OBJDIR)/hash: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p hash

clean:
	@echo Cleaning...
	@rm -f obj/*.o
	@rm -f obj/GPU/*.o
	@rm -f obj/hash/*.o
	@rm -f vanitysearch

.PHONY: all clean VanitySearch
