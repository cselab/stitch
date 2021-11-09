#include <inttypes.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void amax0(uint64_t nx, uint64_t ny, uint64_t nz, uint64_t sx, uint64_t sy,
           uint64_t sz, const void *input, uint64_t oy, uint64_t oz,
           void *output) {
  int Verbose;
  Verbose = getenv("STITCH_VERBOSE") != NULL;
  if (Verbose) {
    fprintf(stderr, "stitch.fast.amax0: %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",
            nx, ny, nz);
    fprintf(stderr, "stitch.fast.amax0: %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",
            sx, sy, sz);
    fprintf(stderr, "stitch.fast.amax0: %" PRIu64 " %" PRIu64 "\n", oy, oz);
  }

#pragma omp parallel for
  for (uint64_t k = 0; k < nz; k++) {
    if (Verbose)
      fprintf(stderr, "stitch.fast.amax0 [%d]: %" PRIu64 " / %" PRIu64 "\n",
              omp_get_thread_num(), k + 1, nz);
    for (uint64_t j = 0; j < ny; j++) {
      uint16_t ma;
      ma = 0;
      const void *input0 = input + k * sz + j * sy;
      for (uint64_t i = 0; i < nx; i++) {
        uint16_t x;
        x = *(uint16_t *)(input0 + i * sx);
        if (x > ma)
          ma = x;
      }
      *(uint16_t *)(output + k * oz + j * oy) = ma;
    }
  }
}

void amax1(uint64_t nx, uint64_t ny, uint64_t nz, uint64_t sx, uint64_t sy,
           uint64_t sz, const void *input, uint64_t ox, uint64_t oz,
           void *output) {
  int Verbose;
  Verbose = getenv("STITCH_VERBOSE") != NULL;
  if (Verbose) {
    fprintf(stderr, "stitch.fast.amax0: %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",
            nx, ny, nz);
    fprintf(stderr, "stitch.fast.amax0: %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",
            sx, sy, sz);
    fprintf(stderr, "stitch.fast.amax0: %" PRIu64 " %" PRIu64 "\n", ox, oz);
  }

#pragma omp parallel for
  for (uint64_t k = 0; k < nz; k++) {
    if (Verbose)
      fprintf(stderr, "stitch.fast.amax0 [%d]: %" PRIu64 " / %" PRIu64 "\n",
              omp_get_thread_num(), k + 1, nz);
    for (uint64_t i = 0; i < nx; i++) {
      uint16_t ma;
      ma = 0;
      const void *input0 = input + k * sz + i * sx;
      for (uint64_t j = 0; j < ny; j++) {
        uint16_t x;
        x = *(uint16_t *)(input0 + j * sy);
        if (x > ma)
          ma = x;
      }
      *(uint16_t *)(output + k * oz + i * ox) = ma;
    }
  }
}
