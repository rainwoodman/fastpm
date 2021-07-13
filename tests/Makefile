include ../Makefile.local
include ../Makefile.rules

GSL_DIR   ?=

DIR_PATH = $(GSL_DIR)

CPPFLAGS += $(foreach dir, $(DIR_PATH), -I$(dir)/include)
LDFLAGS += $(foreach dir, $(DIR_PATH), -L$(dir)/lib)

CPPFLAGS += -I../api/ -I../lua/ -I../depends/install/include

TEST_SOURCES = testpm.c \
               testfof.c \
               testcosmology.c \
               testrfof.c \
               testconstrained.c \
               testlightcone.c \
               testangulargrid.c \
               testboxsphere.c

#			   testlightconeP.c

TESTS = $(TEST_SOURCES:%.c=%)

LIBFASTPM_LIBS = ../libfastpmio/libfastpm-io.a ../libfastpm/libfastpm.a ../depends/libfastpm-dep.a

SOURCES = $(TEST_SOURCES)
.PHONY: all

all: $(TESTS)

testpm : .objs/testpm.o $(LIBFASTPM_LIBS)
	$(CC) $(CPPFLAGS) $(OPTIMIZE) $(OPENMP) -o $@ $^ \
        $(LDFLAGS) $(GSL_LIBS) -lm

testcosmology : .objs/testcosmology.o $(LIBFASTPM_LIBS)
	$(CC) $(CPPFLAGS) $(OPTIMIZE) $(OPENMP) -o $@ $^ \
	    $(LDFLAGS) $(GSL_LIBS) -lm

testfof : .objs/testfof.o $(LIBFASTPM_LIBS)
	$(CC) $(CPPFLAGS) $(OPTIMIZE) $(OPENMP) -o $@ $^ \
	    $(LDFLAGS) $(GSL_LIBS) -lm

testrfof : .objs/testrfof.o $(LIBFASTPM_LIBS)
	$(CC) $(CPPFLAGS) $(OPTIMIZE) $(OPENMP) -o $@ $^ \
	    $(LDFLAGS) $(GSL_LIBS) -lm

testconstrained : .objs/testconstrained.o $(LIBFASTPM_LIBS)
	$(CC) $(CPPFLAGS) $(OPTIMIZE) $(OPENMP) -o $@ $^ \
	    $(LDFLAGS) $(GSL_LIBS) -lm

testlightcone : .objs/testlightcone.o $(LIBFASTPM_LIBS)
	$(CC) $(CPPFLAGS) $(OPTIMIZE) $(OPENMP) -o $@ $^ \
	    $(LDFLAGS) $(GSL_LIBS) -lm

testboxsphere: .objs/testboxsphere.o $(LIBFASTPM_LIBS)
	$(CC) $(CPPFLAGS) $(OPTIMIZE) $(OPENMP) -o $@ $^ \
	    $(LDFLAGS) $(GSL_LIBS) -lm

testangulargrid: .objs/testangulargrid.o $(LIBFASTPM_LIBS)
	$(CC) $(CPPFLAGS) $(OPTIMIZE) $(OPENMP) -o $@ $^ \
	    $(LDFLAGS) $(GSL_LIBS) -lm

testlightconeP : .objs/testlightconeP.o $(LIBFASTPM_LIBS)
		$(CC) $(OPTIMIZE) $(OPENMP) -o $@ $^ \
				$(LDFLAGS) $(GSL_LIBS) -lm

testrecorder: .objs/testrecorder.o $(LIBFASTPM_LIBS)
	$(CC) $(CPPFLAGS) $(OPTIMIZE) $(OPENMP) -o $@ $^ \
	    $(LDFLAGS) $(GSL_LIBS) -lm


-include $(SOURCES:%.c=.deps/%.d)

clean:
	rm -rf .objs
	rm -rf .deps
	rm -rf $(TESTS)
