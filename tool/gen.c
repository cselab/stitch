#define POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 500

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PI (3.141592653589793)
#define	USED(x)		if(x);else{}
enum { X, Y, Z };
static const char *me = "adv.gen";
static void int16_to_little(uint16_t, unsigned char *, unsigned char *);
static void
usg(void)
{
    fprintf(stderr, "%s -n int int int -o int int [-d directory] [-w float]\n", me);
    exit(2);
}

static int shift[2][2][3] = {
    { { 0, 0, 0}, { 0, 0, 0} },
    { { 0, 0, 0}, { 0, 0, 0} },
};

static uint16_t
pattern0(double x, double y, double z)
{
    double rsq;

    x -= 0.4;
    y -= 0.6;
    z -= 0.25;
    rsq = x * x + y * y + z * z;
    return 0.2 * 0.2 < rsq && rsq < 0.45 * 0.45 ? 100 : 0;
}

static uint16_t
pattern1(double x, double y, double z)
{
    int i;
    double rsq;
    double u;
    double v;
    double w;
    const double r[][3] = { {0.53,0.42,0.95},
			    {0.07,0.29,0.50},
			    {0.29,0.66,0.59},
			    {0.60,0.09,0.99},
			    {0.59,0.04,0.46},
			    {0.59,0.97,0.24},
			    {0.32,0.04,0.82},
			    {0.18,0.70,0.47},
			    {0.95,0.11,0.54},
			    {0.82,0.80,0.03},
			    {0.82,0.55,0.57},
			    {0.86,0.24,0.64},
    };
    for (i = 0; i < sizeof r / sizeof *r; i++) {
      u = x - r[i][0];
      v = y - r[i][1];
      w = z - r[i][2];
      rsq = u*u + v*v + w*w;
      if (rsq < 0.3 * 0.3)
	return 100;
    }
    return 0;
}

static int tx = sizeof shift / sizeof shift[0];
static int ty = sizeof shift[0] / sizeof shift[0][0];

int
main(int argc, char **argv)
{
    char cmd[FILENAME_MAX];
    char *end;
    char mhd[FILENAME_MAX];
    char raw_filename[FILENAME_MAX];
    char raw_path[FILENAME_MAX];
    char xdmf[FILENAME_MAX];
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
    int x;
    int y;
    int wx;
    int wy;
    uint16_t value;
    unsigned char a;
    unsigned char b;
    uint16_t (*pattern)(double, double, double);

    USED(argc);
    Oflag = 0;
    Nflag = 0;
    w = 0;
    pattern = pattern0;
    while (*++argv != NULL && argv[0][0] == '-')
        switch (argv[0][1]) {
        case 'h':
            usg();
            break;
        case 'd':
            argv++;
            if (argv[0] == NULL) {
                fprintf(stderr, "%s: -d needs an argument\n", me);
                exit(2);
            }
            directory = *argv;
            break;
        case 'o':
            argv++;
            if (argv[0] == NULL || argv[1] == NULL) {
                fprintf(stderr, "%s: -o needs two arguments\n", me);
                exit(2);
            }
            overlap[X] = strtol(*argv, &end, 10);
            if (errno != 0 || *end != '\0') {
                fprintf(stderr, "%s: not an integer '%s'\n", me, *argv);
                exit(2);
            }
            argv++;
            overlap[Y] = strtol(*argv, &end, 10);
            if (errno != 0 || *end != '\0') {
                fprintf(stderr, "%s: not an integer '%s'\n", me, *argv);
                exit(2);
            }
            Oflag = 1;
            break;
        case 'n':
            argv++;
            if (argv[0] == NULL || argv[1] == NULL || argv[2] == NULL) {
                fprintf(stderr, "%s: -n needs three arguments\n", me);
                exit(2);
            }
            nx = strtol(*argv, &end, 10);
            if (errno != 0 || *end != '\0') {
                fprintf(stderr, "%s: not an integer '%s'\n", me, *argv);
                exit(2);
            }
            argv++;
            ny = strtol(*argv, &end, 10);
            if (errno != 0 || *end != '\0') {
                fprintf(stderr, "%s: not an integer '%s'\n", me, *argv);
                exit(2);
            }
            argv++;
            nz = strtol(*argv, &end, 10);
            if (errno != 0 || *end != '\0') {
                fprintf(stderr, "%s: not an integer '%s'\n", me, *argv);
                exit(2);
            }
            Nflag = 1;
            break;
        case 'w':
            argv++;
            if (argv[0] == NULL) {
                fprintf(stderr, "%s: -w needs an argument\n", me);
                exit(2);
            }
            w = strtod(*argv, &end);
            if (errno != 0 || *end != '\0') {
                fprintf(stderr, "%s: not a float '%s'\n", me, *argv);
                exit(2);
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
    snprintf(cmd, sizeof cmd, "mkdir -p '%s' 2>/dev/null", directory);
    if (system(cmd) != 0) {
        fprintf(stderr, "%s: command <%s> failed\n", me, cmd);
        exit(1);
    }
    snprintf(cmd, sizeof cmd, "test -d '%s'", directory);
    if (system(cmd) != 0) {
        fprintf(stderr, "%s: failed to create directory '%s'\n", me,
                directory);
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
	        wx = w * sin(4 * PI / nz * k);
		wy = w * cos(4 * PI / nz * k);
                for (j = 0; j < ny; j++)
                    for (i = 0; i < nx; i++) {
                        ix = i + x * (nx - overlap[X]) - overlap[X] -
                            shift[x][y][X] + wx;
                        iy = j + y * (ny - overlap[Y]) - overlap[Y] -
                            shift[x][y][Y] + wy;
                        iz = k - shift[x][y][Z];
                        value =
			  pattern(scale * ix, scale * iy, scale * iz);
                        int16_to_little(value, &a, &b);
                        if (fputc(a, file) == EOF || fputc(b, file) == EOF) {
                            fprintf(stderr, "%s: fail to wirte '%s'\n", me,
                                    raw_path);
                            exit(2);
                        }
                    }
	    }
        }
}

static void
int16_to_little(uint16_t x, unsigned char *a, unsigned char *b)
{
    *a = x & 0xFF;
    *b = x >> 8;
}
