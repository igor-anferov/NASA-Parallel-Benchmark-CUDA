SHELL=/bin/sh
BENCHMARK=sp
BENCHMARKU=SP
DEVICE_CC=nvcc
ARCH=sm_60

include ../config/make.def

include ../sys/make.common

OBJS = sp.o allocate.o initialize.o exact_solution.o exact_rhs.o \
       set_constants.o adi.o rhs.o      \
       x_solve.o ninvr.o y_solve.o pinvr.o    \
       z_solve.o tzetar.o add.o txinvr.o error.o verify.o deallocate.o  \
       ${COMMON}/print_results.o ${COMMON}/c_timers.o ${COMMON}/wtime.o

CUDA_OBJS = cuda_init.cpp.o cuda_memory.cpp.o initialize.cu.o rhs.cu.o txinvr.cu.o add.cu.o x_solve.cu.o ninvr.cu.o y_solve.cu.o pinvr.cu.o z_solve.cu.o tzetar.cu.o

ifdef NEED_CUDA
DEVICE_CC += -DNEED_CUDA
OBJS += ${CUDA_OBJS}
endif

# npbparams.h is included by header.h
# The following rule should do the trick but many make programs (not gmake)
# will do the wrong thing and rebuild the world every time (because the
# mod time on header.h is not changed. One solution would be to 
# touch header.h but this might cause confusion if someone has
# accidentally deleted it. Instead, make the dependency on npbparams.h
# explicit in all the lines below (even though dependence is indirect). 

# header.h: npbparams.h

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

%.cu.o:: %.cu  header.h npbparams.h
	${DEVICE_CC} -Xcompiler -rdynamic -lineinfo -g -O3 -arch ${ARCH} ${C_INC} $< -c -o $@

%.o:: %.c  header.h npbparams.h
	${CCOMPILE} $< -o $@

%.cpp.o:: %.cpp  header.h npbparams.h
	${CCOMPILE_pp} $< -o $@

clean:
	- rm -f *.o *~ mputil*
	- rm -f npbparams.h core
