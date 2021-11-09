.POSIX:
.SUFFIXES:
PY = python3
CC = c99
CFLAGS = -O3

all: install
M = \
stitch/fast.py \
stitch/glb.py \
stitch/Rigid.py \
stitch/stitch0.so \
stitch/Tracking.py \
stitch/union_find.py \
stitch/Wobbly.py \

install: $M
	@p=`"$(PY)" -m site --user-site` || exit 2 && \
	mkdir -p "$$p/stitch" && \
	for i in $M; do cp -- "$$i" "$$p/stitch/$$f" || exit 2; done && \
	printf '%s\n' "$$p"

stitch/stitch0.so: stitch/stitch.c
	$(CC) $(CFLAGS) -fopenmp -shared -fPIC -o $@ $< $(LDFLAGS)
clean:; rm -f stitch/stitch0.so
