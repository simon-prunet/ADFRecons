export CPP := g++
export CPPFLAGS := -O2 -Wno-write-strings
export FC := gfortran -O2
export FFLAGS := -fno-second-underscore -w -fallow-argument-mismatch
LIBS := -lm -lc -Llib -lport_i -lfittools

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LIBS += -lgfortran
endif
ifeq ($(UNAME_S),Darwin)
    LIBS += -L/usr/local/Cellar/gcc/11.3.0_2/lib/gcc/11/ -lgfortran
endif


TARGETS := bin/recons

.PHONY: all clean

all: $(TARGETS)

clean:
	@$(MAKE) -C port_i clean
	@rm -rf bin lib objs $(TARGETS)

bin/%: objs/%.o lib/libfittools.a
	@mkdir -p bin
	@$(CPP) $(CPPFLAGS) -o $@ $< lib/libfittools.a $(LIBS)
	@rm -f $<

objs/%.o: src/%.cxx src/%.h objs
	@$(CPP) $(CPPFLAGS) -o $@ -c $<

objs/%.o: src/%.cxx objs
	@$(CPP) $(CPPFLAGS) -o $@ -c $<

objs/%.o: src/%.f objs
	@$(FC) $(FFLAGS) -o $@ -c $<

lib/libfittools.a: objs/FitTools.o objs/AS153.o lib/libport_i.a
	@mkdir -p lib
	@ar rv lib/libfittools.a objs/FitTools.o objs/AS153.o
	@ranlib lib/libfittools.a

lib/libport_i.a:
	@mkdir -p lib
	@$(MAKE) -e -C port_i
	@cd lib && ln -s ../port_i/lib/libport_i.a libport_i.a

objs:
	@mkdir -p objs
