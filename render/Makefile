
EXECUTABLE := render

CU_FILES   := cudaRenderer.cu

CU_DEPS    :=

CC_FILES   := main.cpp display.cpp benchmark.cpp refRenderer.cpp \
              noise.cpp ppm.cpp sceneLoader.cpp

LOGS	   := logs

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=nvcc -m64 -w
CXXFLAGS=-O3 -g -I"D:\Program Files (x86)\NVIDIA Corporation\CUDA\common\inc"
HOSTNAME=$(shell hostname)

LIBS       :=
FRAMEWORKS :=

#TODO switch to 61 and update ref
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_35
LIBS += freeglut glew64 cudart

LDFLAGS=-L"D://Program Files (x86)//NVIDIA GPU Computing//lib//x64" -lcudart \
	-L"D:\Program Files (x86)\NVIDIA Corporation\CUDA\common\lib\x64"

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc -w

OBJS=$(OBJDIR)/main.obj $(OBJDIR)/display.obj $(OBJDIR)/benchmark.obj $(OBJDIR)/refRenderer.obj \
     $(OBJDIR)/cudaRenderer.obj $(OBJDIR)/noise.obj $(OBJDIR)/ppm.obj $(OBJDIR)/sceneLoader.obj \
     $(OBJDIR)/getopt.obj $(OBJDIR)/getopt1.obj


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir $(OBJDIR)

clean:
		del /Q $(EXECUTABLE).exe $(LOGS)
		rmdir /Q /S $(OBJDIR)

check:	default
		./checker.pl

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

$(OBJDIR)/%.obj: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.obj: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@

$(OBJDIR)/%.obj: %.c
		$(CXX) $< $(CXXFLAGS) -c -o $@
