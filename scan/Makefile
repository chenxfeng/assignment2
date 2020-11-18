EXECUTABLE := cudaScan

CU_FILES   := scan.cu

CU_DEPS    :=

CC_FILES   := main.cpp

all: $(EXECUTABLE) $(REFERENCE)

LOGS	   := logs

###########################################################

OBJDIR=objs
CXX=nvcc -m64 -w
CXXFLAGS=-O3
LDFLAGS=-L"D://Program Files (x86)//NVIDIA GPU Computing//lib//x64" -lcudart
NVCC=nvcc -w
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_35


OBJS=$(OBJDIR)/main.obj  $(OBJDIR)/scan.obj  $(OBJDIR)/getopt.obj $(OBJDIR)/getopt1.obj


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir $(OBJDIR)

clean:
		del /Q *.ppm $(EXECUTABLE).exe $(LOGS)
		rmdir /Q /S $(OBJDIR)

check_scan: default
				./checker.pl scan

check_find_repeats: default
				./checker.pl find_repeats

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.obj: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.obj: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@

$(OBJDIR)/%.obj: %.c
		$(CXX) $< $(CXXFLAGS) -c -o $@
