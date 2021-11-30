.POSIX:
.SUFFIXES:
PY = python3
CC = c99
CFLAGS = -O3
CFLAGS_OPENMP = -fopenmp
CFLAGS_FPIC = -fPIC

all: install
M = \
stitch/fast.py \
stitch/glb.py \
stitch/mesospim.py\
stitch/Rigid.py \
stitch/stitch0.so \
stitch/Tracking.py \
stitch/union_find.py \
stitch/Wobbly.py \

install: $M
	@p=`"$(PY)" -m site --user-site` || exit 1 && \
	mkdir -p "$$p/stitch" && \
	for i in $M; do cp -- "$$i" "$$p/$$i" || exit 1; done && \
	printf '%s\n' "$$p"

stitch/stitch0.so: stitch/stitch.c
	$(CC) $(CFLAGS) $(CFLAGS_OPENMP) $(CFLAGS_FPIC) -shared -o $@ $< $(LDFLAGS)
uninstall:
	p=`"$(PY)" -m site --user-site` || exit 1 && \
	for i in $M; do rm -- "$$p/$$i" || exit 1; done && \
	rmdir -- "$$p/stitch"
clean:; rm -f stitch/stitch0.so
