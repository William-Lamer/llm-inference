#include <errno.h>
#include <string.h>
#include <sys/stat.h>

#include "dump.cuh"

#define MAX_DUMP_NAMES 1024
#define MAX_NAME_LEN 64

static const char *dump_dir = NULL;
static char seen_names[MAX_DUMP_NAMES][MAX_NAME_LEN];
static int n_seen = 0;

void dump_open(const char *dir) {
    if (mkdir(dir, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "dump: cannot create %s: %s\n", dir, strerror(errno));
        exit(EXIT_FAILURE);
    }
    dump_dir = dir;
}

// The first write of a name in this run truncates the file, after it appends.
static int seen_before(const char *name) {
    for (int i = 0; i < n_seen; i++) {
        if (strcmp(seen_names[i], name) == 0) {
            return 1;
        }
    }
    if (n_seen == MAX_DUMP_NAMES || strlen(name) >= MAX_NAME_LEN) {
        fprintf(stderr, "dump: too many names or name too long: %s\n", name);
        exit(EXIT_FAILURE);
    }
    strcpy(seen_names[n_seen], name);
    n_seen++;
    return 0;
}

static void write_row(const char *name, const float *row, int n) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", dump_dir, name);
    const char *mode = seen_before(name) ? "ab" : "wb";
    FILE *f = fopen(path, mode);
    if (f == NULL) {
        fprintf(stderr, "dump: cannot open %s\n", path);
        exit(EXIT_FAILURE);
    }
    if (fwrite(row, sizeof(float), n, f) != (size_t)n) {
        fprintf(stderr, "dump: cannot write %s\n", path);
        exit(EXIT_FAILURE);
    }
    fclose(f);
}

// cudaMemcpy waits for every kernel launched before it so the dump sees the finished tensor.
void dump_bf16(const char *name, const bf16 *d_data, int n) {
    if (dump_dir == NULL) {
        return;
    }
    bf16 *host = (bf16 *)malloc(n * sizeof(bf16));
    float *row = (float *)malloc(n * sizeof(float));
    if (host == NULL || row == NULL) {
        fprintf(stderr, "dump: out of memory\n");
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemcpy(host, d_data, n * sizeof(bf16), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) row[i] = __bfloat162float(host[i]);
    write_row(name, row, n);
    free(host);
    free(row);
}

void dump_f32(const char *name, const float *d_data, int n) {
    if (dump_dir == NULL) {
        return;
    }
    float *row = (float *)malloc(n * sizeof(float));
    if (row == NULL) {
        fprintf(stderr, "dump: out of memory\n");
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemcpy(row, d_data, n * sizeof(float), cudaMemcpyDeviceToHost));
    write_row(name, row, n);
    free(row);
}
