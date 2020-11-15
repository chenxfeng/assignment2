
EXECUTABLE := cudaSaxpy

CU_FILES   := saxpy.cu

CU_DEPS    :=

CC_FILES   := main.cpp

###########################################################

# ARCH=$(shell uname | sed -e 's/-.*//g')

# nvcc compiler in fact call the VisualStudio's cl compiler
OBJDIR=objs
CXX=nvcc -m64
CXXFLAGS=-O3
LDFLAGS=-L"D://Program Files (x86)//NVIDIA GPU Computing//lib//x64" -lcudart
NVCC=nvcc -w
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_35


OBJS=$(OBJDIR)/main.obj  $(OBJDIR)/saxpy.obj


.PHONY: dirs clean

default: $(EXECUTABLE)

# mkdir -p $(OBJDIR)/
dirs:
		mkdir $(OBJDIR)

#rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)
clean:
		del /Q *.ppm $(EXECUTABLE).exe
		rmdir /Q /S $(OBJDIR)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.obj: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.obj: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
