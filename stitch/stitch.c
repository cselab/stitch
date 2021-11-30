#include <inttypes.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_thread_num(void) { return 0; }
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint16_t type;
typedef uint64_t type_sum;
#define TYPE_MIN (0)

int amax0(uint64_t nx, uint64_t ny, uint64_t nz, uint64_t sx, uint64_t sy,
          uint64_t sz, const uint8_t *input, uint64_t oy, uint64_t oz,
          uint8_t *output) {
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
      type ma;
      ma = TYPE_MIN;
      const uint8_t *input0 = input + k * sz + j * sy;
      for (uint64_t i = 0; i < nx; i++) {
        type x;
        x = *(type *)(input0 + i * sx);
        if (x > ma)
          ma = x;
      }
      *(type *)(output + k * oz + j * oy) = ma;
    }
  }
  return 0;
}

int amax1(uint64_t nx, uint64_t ny, uint64_t nz, uint64_t sx, uint64_t sy,
          uint64_t sz, const uint8_t *input, uint64_t ox, uint64_t oz,
          uint8_t *output) {
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
      type ma;
      ma = TYPE_MIN;
      const uint8_t *input0 = input + k * sz + i * sx;
      for (uint64_t j = 0; j < ny; j++) {
        type x;
        x = *(type *)(input0 + j * sy);
        if (x > ma)
          ma = x;
      }
      *(type *)(output + k * oz + i * ox) = ma;
    }
  }
  return 0;
}

double corr(uint64_t nx, uint64_t ny, uint64_t nz, uint64_t sx0, uint64_t sy0,
            uint64_t sz0, const uint8_t *a, uint64_t sx1, uint64_t sy1,
            uint64_t sz1, const uint8_t *b) {
  int Verbose;
  type_sum s_xx;
  type_sum s_yy;
  type_sum s_xy;
  type_sum s_x;
  type_sum s_y;
  type_sum n;

  double d_xx;
  double d_yy;
  double d_xy;
  double d_x;
  double d_y;
  double v_x;
  double v_y;
  double v_xy;

  Verbose = getenv("STITCH_VERBOSE") != NULL;
  if (Verbose) {
    fprintf(stderr, "stitch.corr: %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", nx,
            ny, nz);
    fprintf(stderr, "stitch.corr: %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", sx0,
            sy0, sz0);
    fprintf(stderr, "stitch.corr: %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", sx1,
            sy1, sz1);
  }
  s_xx = s_xy = s_yy = s_x = s_y = 0;
#pragma omp parallel for
  for (uint64_t k = 0; k < nz; k++) {
    if (Verbose)
      fprintf(stderr, "stitch.corr [%d]: %" PRIu64 " / %" PRIu64 "\n",
              omp_get_thread_num(), k + 1, nz);
    for (uint64_t j = 0; j < ny; j++) {
      const uint8_t *a0 = a + k * sz0 + j * sy0;
      const uint8_t *b0 = b + k * sz1 + j * sy1;
      for (uint64_t i = 0; i < nx; i++) {
        type x;
        type y;
        type_sum xx;
        type_sum yy;
        type_sum xy;
        x = (type_sum) * (type *)(a0 + i * sx0);
        y = (type_sum) * (type *)(b0 + i * sx1);
        xx = x * x;
        yy = y * y;
        xy = x * y;
#pragma omp atomic
        s_xx += xx;
#pragma omp atomic
        s_yy += yy;
#pragma omp atomic
        s_xy += xy;
#pragma omp atomic
        s_x += x;
#pragma omp atomic
        s_y += y;
      }
    }
  }

  d_xx = s_xx;
  d_xy = s_xy;
  d_yy = s_yy;
  d_x = s_x;
  d_y = s_y;
  n = nx * ny * nz;

  v_xy = n * d_xy - d_x * d_y;
  v_x = n * d_xx - d_x * d_x;
  v_y = n * d_yy - d_y * d_y;

  return (v_x == 0 || v_y == 0) ? 0 : v_xy / sqrt(v_x * v_y);
}
