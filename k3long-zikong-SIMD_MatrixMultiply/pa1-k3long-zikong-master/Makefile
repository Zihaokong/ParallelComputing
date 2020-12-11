# This makefile is intended for the GNU C compiler.
# Your code must compile (with GCC) with the given CFLAGS.
# You may experiment with the MY_OPT variable to invoke additional compiler options
HOST = $(shell hostname)
BANG = $(shell hostname | grep ccom-bang | wc -c)
BANG-COMPUTE = $(shell hostname | grep compute | wc -c)
AMAZON = $(shell hostname | grep 'ip-' | wc -c)
WSL =  $(shell uname -a | grep -i '\-microsoft' | wc -c)

ifneq ($(BANG), 0)
atlas := 1
multi := 0
NO_BLAS = 1
include $(PUB)/Arch/arch.gnu_c99.generic
else
ifneq ($(BANG-COMPUTE), 0)
atlas := 1
multi := 0
NO_BLAS = 1
include $(PUB)/Arch/arch.gnu_c99.generic
else
ifneq ($(AMAZON), 0)
atlas := 0
openblas := 1
multi := 0
NO_BLAS = 1
include $(PUB)/Arch/arch.gnu_c99.generic
CFLAGS += -mfma
MY_OPT = "-O3"
MY_OPT += "-march=core-avx2"
MY_OPT += "-DOPENBLAS_SINGLETHREAD"
else
ifneq ($(WSL), 0)
PUB=~/cse260fa20/pa/pub/cse260-fa20
atlas := 0
openblas := 1
multi := 0
NO_BLAS = 1
include $(PUB)/Arch/arch.gnu_c99.generic
ifeq ($(debug), 1)
	MY_OPT = "-O0"
else
	MY_OPT = "-O3"
endif
MY_OPT += "-march=core-avx2"
MY_OPT += "-DOPENBLAS_SINGLETHREAD"
# MY_OPT += "-fPIC"
# MY_OPT = "-O3"
# MY_OPT = "-O4"
endif
endif
endif
endif

# WARNINGS += -Wall -pedantic
WARNINGS += -Wall 

# If you want to copy data blocks to contiguous storage
# This applies to the hand code version
ifeq ($(copy), 1)
    C++FLAGS += -DCOPY
    CFLAGS += -DCOPY
endif


# If you want to use restrict pointers, make restrict=1
# This applies to the hand code version
ifeq ($(restrict), 1)
    C++FLAGS += -D__RESTRICT
    CFLAGS += -D__RESTRICT
# ifneq ($(CARVER), 0)
#    C++FLAGS += -restrict
#     CFLAGS += -restrict
# endif
endif

ifeq ($(NO_BLAS), 1)
    C++FLAGS += -DNO_BLAS
    CFLAGS += -DNO_BLAS
endif

ifeq ($(debug), 1)
	CFLAGS += -g -DDEBUG
endif

OPTIMIZATION = $(MY_OPT)

targets = benchmark-naive benchmark-blas benchmark-blislab
BLISLAB = blislab/bl_dgemm_ukr.c  blislab/my_dgemm.c blislab/bl_dgemm_util.c
objects = benchmark.o \
	bl_dgemm_util.o \
	blas/dgemm-blas.o \
	naive/dgemm-naive.o \
	$(BLISLAB:.c=.o)

UTIL   = wall_time.o cmdLine.o debugMat.o blislab/bl_dgemm_util.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o naive/dgemm-naive.o  $(UTIL)
	$(CC) -o $@ $^ $(LDLIBS) 
benchmark-blas : benchmark.o blas/dgemm-blas.o $(UTIL)
	$(CC) -o $@ $^ $(LDLIBS) -static
benchmark-blislab : $(BLISLAB:.c=.o) benchmark.o $(UTIL) 
	$(CC) -o $@ $^ $(LDLIBS) -pg

%.o : %.c
	$(CC) -c $(CFLAGS) -c $< -o $@ $(OPTIMIZATION) 


%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@ $(OPTIMIZATION)



.PHONY : clean
clean:
	rm -f $(targets) $(objects) $(UTIL) core

