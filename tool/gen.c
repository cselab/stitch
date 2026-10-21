#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PI (3.141592653589793)
#define USED(x)                                                                \
  if (x)                                                                       \
    ;                                                                          \
  else {                                                                       \
  }
enum { X, Y, Z };
static const char *me = "gen";
static void int16_to_little(uint16_t, unsigned char *, unsigned char *);
static void usg(void) {
  fprintf(stderr,
          "%s -p 1|2|3 -n int int int -o int int [-d directory] [-w float]\n",
          me);
  exit(2);
}

static int shift[2][2][3] = {
    {{1, 2, 3}, {0, 0, 0}},
    {{0, 0, 0}, {1, 2, 3}},
};

static uint16_t pattern1(double x, double y, double z) {
  double rsq;

  x -= 0.4;
  y -= 0.6;
  z -= 0.25;
  rsq = x * x + y * y + z * z;
  return 0.2 * 0.2 < rsq && rsq < 0.45 * 0.45 ? 100 : 0;
}

static uint16_t pattern2(double x, double y, double z) {
  int i;
  double rsq;
  double u;
  double v;
  double w;
  const double r[][3] = {
      {0.53, 0.42, 0.95}, {0.07, 0.29, 0.50}, {0.29, 0.66, 0.59},
      {0.60, 0.09, 0.99}, {0.59, 0.04, 0.46}, {0.59, 0.97, 0.24},
      {0.32, 0.04, 0.82}, {0.18, 0.70, 0.47}, {0.95, 0.11, 0.54},
      {0.82, 0.80, 0.03}, {0.82, 0.55, 0.57}, {0.86, 0.24, 0.64},
  };
  for (i = 0; i < sizeof r / sizeof *r; i++) {
    u = x - r[i][0];
    v = y - r[i][1];
    w = z - r[i][2];
    rsq = u * u + v * v + w * w;
    if (rsq < 0.3 * 0.3)
      return 100;
  }
  return 0;
}

static double rnd(unsigned long *seed) {
  *seed = *seed * 6364136223846793005UL + 1442695040888963407UL;
  return (*seed >> 11) / 9007199254740992.0;
}

static uint16_t pattern3(double x, double y, double z) {
  enum { N = 300 };
  static double c[N][7];
  static int init = 0;
  unsigned long seed;
  int i;
  double d;
  double n;
  double rsq;
  double s;
  double t;
  double u;
  double v;
  double w;

  if (!init) {
    seed = 12345;
    for (i = 0; i < N; i++) {
      c[i][0] = rnd(&seed);
      c[i][1] = rnd(&seed);
      c[i][2] = rnd(&seed);
      u = 2 * rnd(&seed) - 1;
      v = 2 * rnd(&seed) - 1;
      w = 2 * rnd(&seed) - 1;
      n = sqrt(u * u + v * v + w * w);
      c[i][3] = u / n;
      c[i][4] = v / n;
      c[i][5] = w / n;
      c[i][6] = 0.002 + 0.004 * rnd(&seed);
    }
    init = 1;
  }
  s = 0;
  for (i = 0; i < N; i++) {
    u = x - c[i][0];
    v = y - c[i][1];
    w = z - c[i][2];
    t = u * c[i][3] + v * c[i][4] + w * c[i][5];
    rsq = u * u + v * v + w * w - t * t;
    d = c[i][6];
    if (rsq < 9 * d * d)
      s += exp(-rsq / (2 * d * d));
  }
  return s > 1 ? 1000 : 1000 * s;
}

static int tx = sizeof shift / sizeof shift[0];
static int ty = sizeof shift[0] / sizeof shift[0][0];

static const struct {
  const char *pattern;
  uint16_t (*function)(double, double, double);
} Pattern[3] = {
    {"1", pattern1},
    {"2", pattern2},
    {"3", pattern3},
};

int main(int argc, char **argv) {
  char cmd[FILENAME_MAX];
  char *end;
  char raw_path[FILENAME_MAX];
  const char *directory = ".";
  double ix;
  double iy;
  double iz;
  double scale;
  double w;
  FILE *file;
  int i;
  int j;
  int k;
  int Nflag;
  int nx;
  int ny;
  int nz;
  int Oflag;
  int overlap[2];
  int wx;
  int wy;
  int x;
  int y;
  uint16_t (*pattern)(double, double, double);
  uint16_t value;
  unsigned char a;
  unsigned char b;

  USED(argc);
  Oflag = 0;
  Nflag = 0;
  w = 0;
  pattern = NULL;
  while (*++argv != NULL && argv[0][0] == '-')
    switch (argv[0][1]) {
    case 'h':
      usg();
      break;
    case 'd':
      argv++;
      if (argv[0] == NULL) {
        fprintf(stderr, "%s: -d needs an argument\n", me);
        exit(1);
      }
      directory = *argv;
      break;
    case 'p':
      argv++;
      if (argv[0] == NULL) {
        fprintf(stderr, "%s: -p needs an argument\n", me);
        exit(1);
      }
      for (i = 0;; i++) {
        if (i == sizeof Pattern / sizeof *Pattern) {
          fprintf(stderr, "%s: wrong pattern '%s'\n", me, argv[0]);
          exit(1);
        }
        if (strcmp(argv[0], Pattern[i].pattern) == 0) {
          pattern = Pattern[i].function;
          break;
        }
      }
      break;
    case 'o':
      argv++;
      if (argv[0] == NULL || argv[1] == NULL) {
        fprintf(stderr, "%s: -o needs two arguments\n", me);
        exit(1);
      }
      overlap[X] = strtol(*argv, &end, 10);
      if (errno != 0 || *end != '\0') {
        fprintf(stderr, "%s: not an integer '%s'\n", me, *argv);
        exit(1);
      }
      argv++;
      overlap[Y] = strtol(*argv, &end, 10);
      if (errno != 0 || *end != '\0') {
        fprintf(stderr, "%s: not an integer '%s'\n", me, *argv);
        exit(1);
      }
      Oflag = 1;
      break;
    case 'n':
      argv++;
      if (argv[0] == NULL || argv[1] == NULL || argv[2] == NULL) {
        fprintf(stderr, "%s: -n needs three arguments\n", me);
        exit(1);
      }
      nx = strtol(*argv, &end, 10);
      if (errno != 0 || *end != '\0') {
        fprintf(stderr, "%s: not an integer '%s'\n", me, *argv);
        exit(1);
      }
      argv++;
      ny = strtol(*argv, &end, 10);
      if (errno != 0 || *end != '\0') {
        fprintf(stderr, "%s: not an integer '%s'\n", me, *argv);
        exit(1);
      }
      argv++;
      nz = strtol(*argv, &end, 10);
      if (errno != 0 || *end != '\0') {
        fprintf(stderr, "%s: not an integer '%s'\n", me, *argv);
        exit(1);
      }
      Nflag = 1;
      break;
    case 'w':
      argv++;
      if (argv[0] == NULL) {
        fprintf(stderr, "%s: -w needs an argument\n", me);
        exit(1);
      }
      w = strtod(*argv, &end);
      if (errno != 0 || *end != '\0') {
        fprintf(stderr, "%s: not a float '%s'\n", me, *argv);
        exit(1);
      }
      break;
    default:
      fprintf(stderr, "%s: unknown option '%s'\n", me, argv[0]);
      exit(1);
    }
  if (Oflag == 0) {
    fprintf(stderr, "%s: -o is not set\n", me);
    exit(1);
  }
  if (Nflag == 0) {
    fprintf(stderr, "%s: -n is not set\n", me);
    exit(1);
  }
  if (pattern == NULL) {
    fprintf(stderr, "%s: -p is not set\n", me);
    exit(1);
  }
  snprintf(cmd, sizeof cmd, "mkdir -p '%s' 2>/dev/null", directory);
  if (system(cmd) != 0) {
    fprintf(stderr, "%s: command <%s> failed\n", me, cmd);
    exit(1);
  }
  snprintf(cmd, sizeof cmd, "test -d '%s'", directory);
  if (system(cmd) != 0) {
    fprintf(stderr, "%s: failed to create directory '%s'\n", me, directory);
    exit(1);
  }
  scale = 1.0 / (tx * (nx - overlap[X]));
  for (x = 0; x < tx; x++)
    for (y = 0; y < ty; y++) {
      snprintf(raw_path, sizeof raw_path, "%s/%dx%dx%dle.%02d.%02d.raw",
               directory, nx, ny, nz, x, y);
      if ((file = fopen(raw_path, "w")) == NULL) {
        fprintf(stderr, "%s: fail to wirte '%s'\n", me, raw_path);
        exit(2);
      }
      for (k = 0; k < nz; k++) {
        wx = w * (x + y) / 2 * sin(4 * PI / nz * k + PI / 2 * (2 * x + y));
        wy = w * (x + y) / 2 * cos(4 * PI / nz * k + PI / 2 * (2 * x + y));
        for (j = 0; j < ny; j++)
          for (i = 0; i < nx; i++) {
            ix = i + x * (nx - overlap[X]) - overlap[X] - shift[x][y][X] + wx;
            iy = j + y * (ny - overlap[Y]) - overlap[Y] - shift[x][y][Y] + wy;
            iz = k - shift[x][y][Z];
            value = pattern(scale * ix, scale * iy, scale * iz);
            int16_to_little(value, &a, &b);
            if (fputc(a, file) == EOF || fputc(b, file) == EOF) {
              fprintf(stderr, "%s: fail to wirte '%s'\n", me, raw_path);
              exit(2);
            }
          }
      }
    }
}

static void int16_to_little(uint16_t x, unsigned char *a, unsigned char *b) {
  *a = x & 0xFF;
  *b = x >> 8;
}
