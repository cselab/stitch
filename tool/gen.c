#define POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 500

#include <errno.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#define	USED(x)		if(x);else{}
enum { X, Y, Z };
static const char *me = "adv.gen";
static void int16_to_little(uint16_t, unsigned char *, unsigned char *);
static void
usg(void)
{
    fprintf(stderr, "%s -n int int int -o int int [-d directory]\n", me);
    exit(2);
}

static int shift[2][2][3] = {
    { { 0, 0, 0}, { 0, 0, 0} },
    { { 0, 0, 0}, { 0, 0, 0} },
};

static uint16_t
pattern(double x, double y, double z, void *p)
{
    USED(p);
    double rsq;

    x -= 0.4;
    y -= 0.6;
    z -= 0.25;
    rsq = x * x + y * y + z * z;
    return 0.2 * 0.2 < rsq && rsq < 0.45 * 0.45 ? 100 : 0;
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
    uint16_t value;
    unsigned char a;
    unsigned char b;

    USED(argc);
    Oflag = 0;
    Nflag = 0;
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
            snprintf(raw_filename, sizeof raw_filename, "%02d.%02d.raw", x,
                     y);
            snprintf(raw_path, sizeof raw_path, "%s/%02d.%02d.raw",
                     directory, x, y);
            if ((file = fopen(raw_path, "w")) == NULL) {
                fprintf(stderr, "%s: fail to wirte '%s'\n", me, raw_path);
                exit(2);
            }
            for (k = 0; k < nz; k++)
                for (j = 0; j < ny; j++)
                    for (i = 0; i < nx; i++) {
                        ix = i + x * (nx - overlap[X]) - overlap[X] -
                            shift[x][y][X];
                        iy = j + y * (ny - overlap[Y]) - overlap[Y] -
                            shift[x][y][Y];
                        iz = k - shift[x][y][Z];
                        value =
                            pattern(scale * ix, scale * iy, scale * iz,
                                    NULL);
                        int16_to_little(value, &a, &b);
                        if (fputc(a, file) == EOF || fputc(b, file) == EOF) {
                            fprintf(stderr, "%s: fail to wirte '%s'\n", me,
                                    raw_path);
                            exit(2);
                        }
                    }
            fclose(file);
            snprintf(mhd, sizeof mhd, "%s/%02d.%02d.mhd", directory, x, y);
            if ((file = fopen(mhd, "w")) == NULL) {
                fprintf(stderr, "%s: fail to wirte '%s'\n", me, mhd);
                exit(2);
            }
            fprintf(file,
                    "ObjectType = Image\n"
                    "NDims = 3\n"
                    "DimSize = %d %d %d\n"
                    "ElementType = MET_USHORT\n"
                    "ElementByteOrderMSB = False\n"
                    "ElementDataFile = %s\n", nx, ny, nz, raw_filename);
            fclose(file);
            snprintf(xdmf, sizeof xdmf, "%s/%02d.%02d.xdmf2", directory, x,
                     y);
            if ((file = fopen(xdmf, "w")) == NULL) {
                fprintf(stderr, "%s: fail to wirte '%s'\n", me, xdmf);
                exit(2);
            }
            fprintf(file,
                    "<?xml version=\"1.0\" ?>\n"
                    "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
                    "<Xdmf Version=\"2.0\">\n"
                    " <Domain>\n"
                    "   <Grid Name=\"mesh\" GridType=\"Uniform\">\n"
                    "     <Topology TopologyType=\"3DCORECTMesh\" Dimensions=\"%d %d %d\"/>\n"
                    "     <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n"
                    "       <DataItem Name=\"Origin\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"8\" Format=\"XML\">\n"
                    "         %d %d %d\n"
                    "       </DataItem>\n"
                    "       <DataItem Name=\"Spacing\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n"
                    "         1 1 1\n"
                    "       </DataItem>\n"
                    "     </Geometry>\n"
                    "     <Attribute Name=\"u\" AttributeType=\"Scalar\" Center=\"Cell\">\n"
                    "       <DataItem Dimensions=\"%d %d %d\" NumberType=\"UShort\" Format=\"Binary\">\n"
                    "        %s\n"
                    "       </DataItem>\n"
                    "     </Attribute>\n"
                    "   </Grid>\n"
                    " </Domain>\n"
                    "</Xdmf>\n",
                    nz + 1, ny + 1, nx + 1, 0, y * ny, x * nx, nz, ny, nx,
                    raw_filename);
            fclose(file);
            fprintf(stderr, "%c%s", x != 0 || y != 0 ? ';' : '\'', xdmf);
        }
    fprintf(stderr, "'\n");
}

static void
int16_to_little(uint16_t x, unsigned char *a, unsigned char *b)
{
    *a = x & 0xFF;
    *b = x >> 8;
}
