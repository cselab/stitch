.POSIX:
.SUFFIXES:
.SUFFIXES: .c
CFLAGS = -Ofast -g
M_LDFLAGS = -lm
PREFIX = $(HOME)/.local
0 = \
gen\

all: $0
.c:; $(CC) $< $(CFLAGS) $(LDFLAGS) $(M_LDFLAGS) -o $@
clean:; @-rm $0
