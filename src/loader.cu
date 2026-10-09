#include <dirent.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "cJSON.h"
#include "model.cuh"

typedef struct {
    cJSON *header;
    char *gpu_data;
    size_t data_bytes;
} Shard;

typedef struct {
    Shard shards[MAX_SHARDS];
    int n_shards;
    Weights *w;
    int max_tensors;
} Loader;

static void fatal(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "loader: ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
    exit(EXIT_FAILURE);
}

static char *read_text_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fatal("cannot open %s", path);
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *text = (char *)malloc(size + 1);
    if (text == NULL) {
        fatal("out of memory reading %s", path);
    }
    if (fread(text, 1, size, f) != (size_t)size) {
        fatal("cannot read %s", path);
    }
    text[size] = '\0';
    fclose(f);
    return text;
}

static double json_number(const cJSON *root, const char *key) {
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(root, key);
    if (!cJSON_IsNumber(item)) {
        fatal("config.json has no number \"%s\"", key);
    }
    return item->valuedouble;
}

static int json_int(const cJSON *root, const char *key) {
    return (int)json_number(root, key);
}

void load_config(const char *model_dir, Config *cfg) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);
    char *text = read_text_file(path);
    cJSON *root = cJSON_Parse(text);
    if (root == NULL) {
        fatal("cannot parse %s", path);
    }

    cfg->hidden_size = json_int(root, "hidden_size");
    cfg->intermediate_size = json_int(root, "intermediate_size");
    cfg->n_layers = json_int(root, "num_hidden_layers");
    cfg->n_heads = json_int(root, "num_attention_heads");
    cfg->n_kv_heads = json_int(root, "num_key_value_heads");
    cfg->head_dim = json_int(root, "head_dim");
    cfg->vocab_size = json_int(root, "vocab_size");
    cfg->max_seq_len = json_int(root, "max_position_embeddings");
    cfg->rms_norm_eps = (float)json_number(root, "rms_norm_eps");
    cfg->rope_theta = (float)json_number(root, "rope_theta");

    const cJSON *tie = cJSON_GetObjectItemCaseSensitive(root, "tie_word_embeddings");
    if (!cJSON_IsBool(tie)) {
        fatal("config.json has no bool \"tie_word_embeddings\"");
    }
    cfg->tie_word_embeddings = cJSON_IsTrue(tie);

    if (cfg->n_heads % cfg->n_kv_heads != 0) {
        fatal("num_attention_heads must be a multiple of num_key_value_heads");
    }

    cJSON_Delete(root);
    free(text);
}

// safetensors layout: [u64 header length N][N bytes of JSON][tensor data].
// Header offsets are relative to the start of the tensor data.
static void load_shard(const char *path, Shard *shard) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fatal("cannot open %s", path);
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        fatal("cannot stat %s", path);
    }
    size_t file_size = (size_t)st.st_size;
    void *map = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        fatal("cannot mmap %s", path);
    }
    close(fd);

    if (file_size < 8) {
        fatal("%s: too small to be a safetensors file", path);
    }
    const char *bytes = (const char *)map;
    // The length is stored little-endian (x86).
    uint64_t header_len;
    memcpy(&header_len, bytes, sizeof(header_len));
    if (header_len > file_size - 8) {
        fatal("%s: bad header length", path);
    }

    shard->header = cJSON_ParseWithLength(bytes + 8, header_len);
    if (shard->header == NULL) {
        fatal("%s: cannot parse header", path);
    }

    const char *data = bytes + 8 + header_len;
    shard->data_bytes = file_size - 8 - header_len;
    CUDA_CHECK(cudaMalloc(&shard->gpu_data, shard->data_bytes));
    CUDA_CHECK(cudaMemcpy(shard->gpu_data, data, shard->data_bytes, cudaMemcpyHostToDevice));
    munmap(map, file_size);
}

static int ends_with(const char *s, const char *suffix) {
    size_t s_len = strlen(s);
    size_t suffix_len = strlen(suffix);
    if (s_len < suffix_len) {
        return 0;
    }
    return strcmp(s + s_len - suffix_len, suffix) == 0;
}

static void load_shards(const char *model_dir, Loader *ld) {
    DIR *dir = opendir(model_dir);
    if (dir == NULL) {
        fatal("cannot open directory %s", model_dir);
    }
    for (;;) {
        struct dirent *entry = readdir(dir);
        if (entry == NULL) {
            break;
        }
        if (!ends_with(entry->d_name, ".safetensors")) {
            continue;
        }
        if (ld->n_shards == MAX_SHARDS) {
            fatal("more than %d .safetensors files in %s", MAX_SHARDS, model_dir);
        }
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", model_dir, entry->d_name);
        load_shard(path, &ld->shards[ld->n_shards]);
        ld->n_shards++;
    }
    closedir(dir);
    if (ld->n_shards == 0) {
        fatal("no .safetensors files in %s", model_dir);
    }
}

static int count_tensors(const Shard *shard) {
    int n = cJSON_GetArraySize(shard->header);
    if (cJSON_GetObjectItemCaseSensitive(shard->header, "__metadata__") != NULL) {
        n--;
    }
    return n;
}

static const cJSON *lookup(const Loader *ld, const char *name, const Shard **shard_out) {
    for (int i = 0; i < ld->n_shards; i++) {
        const cJSON *entry = cJSON_GetObjectItemCaseSensitive(ld->shards[i].header, name);
        if (entry != NULL) {
            *shard_out = &ld->shards[i];
            return entry;
        }
    }
    return NULL;
}

// Checks the tensor against the shape the config says and returns its GPU
// address.
static const bf16 *find_tensor(Loader *ld, const char *name, int n_dims, const int *shape) {
    const Shard *shard = NULL;
    const cJSON *entry = lookup(ld, name, &shard);
    if (entry == NULL) {
        fatal("tensor %s not found", name);
    }

    const cJSON *dtype = cJSON_GetObjectItemCaseSensitive(entry, "dtype");
    if (!cJSON_IsString(dtype) || strcmp(dtype->valuestring, "BF16") != 0) {
        fatal("%s: dtype is not BF16", name);
    }

    const cJSON *dims = cJSON_GetObjectItemCaseSensitive(entry, "shape");
    if (cJSON_GetArraySize(dims) != n_dims) {
        fatal("%s: expected %d dims", name, n_dims);
    }
    size_t numel = 1;
    for (int d = 0; d < n_dims; d++) {
        int got = (int)cJSON_GetArrayItem(dims, d)->valuedouble;
        if (got != shape[d]) {
            fatal("%s: dim %d is %d, config.json implies %d", name, d, got, shape[d]);
        }
        numel *= shape[d];
    }

    const cJSON *offsets = cJSON_GetObjectItemCaseSensitive(entry, "data_offsets");
    if (cJSON_GetArraySize(offsets) != 2) {
        fatal("%s: bad data_offsets", name);
    }
    size_t begin = (size_t)cJSON_GetArrayItem(offsets, 0)->valuedouble;
    size_t end = (size_t)cJSON_GetArrayItem(offsets, 1)->valuedouble;
    if (end > shard->data_bytes || end - begin != numel * sizeof(bf16)) {
        fatal("%s: data_offsets do not match its shape", name);
    }
    if (begin % 16 != 0) {
        fatal("%s: offset %zu is not 16-byte aligned", name, begin);
    }

    const bf16 *data = (const bf16 *)(shard->gpu_data + begin);

    Weights *w = ld->w;
    if (w->n_tensors == ld->max_tensors) {
        fatal("%s: loaded more tensors than the files contain", name);
    }
    TensorRef *ref = &w->tensors[w->n_tensors];
    snprintf(ref->name, sizeof(ref->name), "%s", name);
    ref->data = data;
    ref->numel = numel;
    w->n_tensors++;
    return data;
}

static const bf16 *get_vector(Loader *ld, const char *name, int n) {
    int shape[1] = {n};
    return find_tensor(ld, name, 1, shape);
}

static const bf16 *get_matrix(Loader *ld, const char *name, int rows, int cols) {
    int shape[2] = {rows, cols};
    return find_tensor(ld, name, 2, shape);
}

static const bf16 *layer_vector(Loader *ld, int layer, const char *suffix, int n) {
    char name[96];
    snprintf(name, sizeof(name), "model.layers.%d.%s", layer, suffix);
    return get_vector(ld, name, n);
}

static const bf16 *layer_matrix(Loader *ld, int layer, const char *suffix, int rows, int cols) {
    char name[96];
    snprintf(name, sizeof(name), "model.layers.%d.%s", layer, suffix);
    return get_matrix(ld, name, rows, cols);
}

void load_weights(const char *model_dir, const Config *cfg, Weights *w) {
    memset(w, 0, sizeof(*w));
    Loader ld;
    memset(&ld, 0, sizeof(ld));
    ld.w = w;
    load_shards(model_dir, &ld);

    int tensors_in_files = 0;
    for (int i = 0; i < ld.n_shards; i++) {
        tensors_in_files += count_tensors(&ld.shards[i]);
    }
    w->tensors = (TensorRef *)calloc(tensors_in_files, sizeof(TensorRef));
    if (w->tensors == NULL) {
        fatal("out of memory");
    }
    ld.max_tensors = tensors_in_files;

    int hidden = cfg->hidden_size;
    int q_dim = cfg->n_heads * cfg->head_dim;
    int kv_dim = cfg->n_kv_heads * cfg->head_dim;
    int inter = cfg->intermediate_size;
    int head_dim = cfg->head_dim;

    w->embed = get_matrix(&ld, "model.embed_tokens.weight", cfg->vocab_size, hidden);

    w->layers = (LayerWeights *)malloc(cfg->n_layers * sizeof(LayerWeights));
    if (w->layers == NULL) {
        fatal("out of memory");
    }
    for (int l = 0; l < cfg->n_layers; l++) {
        LayerWeights *lw = &w->layers[l];
        lw->attn_norm = layer_vector(&ld, l, "input_layernorm.weight", hidden);
        lw->wq = layer_matrix(&ld, l, "self_attn.q_proj.weight", q_dim, hidden);
        lw->wk = layer_matrix(&ld, l, "self_attn.k_proj.weight", kv_dim, hidden);
        lw->wv = layer_matrix(&ld, l, "self_attn.v_proj.weight", kv_dim, hidden);
        lw->q_norm = layer_vector(&ld, l, "self_attn.q_norm.weight", head_dim);
        lw->k_norm = layer_vector(&ld, l, "self_attn.k_norm.weight", head_dim);
        lw->wo = layer_matrix(&ld, l, "self_attn.o_proj.weight", hidden, q_dim);
        lw->mlp_norm = layer_vector(&ld, l, "post_attention_layernorm.weight", hidden);
        lw->w_gate = layer_matrix(&ld, l, "mlp.gate_proj.weight", inter, hidden);
        lw->w_up = layer_matrix(&ld, l, "mlp.up_proj.weight", inter, hidden);
        lw->w_down = layer_matrix(&ld, l, "mlp.down_proj.weight", hidden, inter);
    }

    w->final_norm = get_vector(&ld, "model.norm.weight", hidden);
    if (cfg->tie_word_embeddings) {
        w->lm_head = w->embed;
    } else {
        w->lm_head = get_matrix(&ld, "lm_head.weight", cfg->vocab_size, hidden);
    }

    const Shard *unused_shard = NULL;
    int allowed_unused = 0;
    if (cfg->tie_word_embeddings && lookup(&ld, "lm_head.weight", &unused_shard) != NULL) {
        allowed_unused = 1;
    }
    int unused = tensors_in_files - w->n_tensors;
    if (unused != allowed_unused) {
        fatal("%d tensors in the safetensors files are not used by the engine", unused);
    }

    for (int i = 0; i < ld.n_shards; i++) {
        w->gpu_buffers[i] = ld.shards[i].gpu_data;
        w->gpu_bytes += ld.shards[i].data_bytes;
        cJSON_Delete(ld.shards[i].header);
    }
    w->n_buffers = ld.n_shards;
}

void free_weights(Weights *w) {
    for (int i = 0; i < w->n_buffers; i++) {
        CUDA_CHECK(cudaFree(w->gpu_buffers[i]));
    }
    free(w->layers);
    free(w->tensors);
    memset(w, 0, sizeof(*w));
}

// Copies each tensor back from the gpu and prints two exact integer checksums
// of its raw bf16 bits: a plain sum, and a position-weighted sum
// that also catches shifted or reordered data.
// ref/weight_checksums.py prints the same from the safetensors files.
void print_weight_checksums(const Weights *w) {
    for (int i = 0; i < w->n_tensors; i++) {
        const TensorRef *t = &w->tensors[i];
        uint16_t *bits = (uint16_t *)malloc(t->numel * sizeof(uint16_t));
        if (bits == NULL) {
            fatal("out of memory");
        }
        CUDA_CHECK(cudaMemcpy(bits, t->data, t->numel * sizeof(uint16_t), cudaMemcpyDeviceToHost));

        uint64_t sum = 0;
        uint64_t weighted = 0;
        for (size_t j = 0; j < t->numel; j++) {
            sum += bits[j];
            weighted += (j + 1) * bits[j];
        }

        // A bf16 is the top 16 bits of an fp32.
        uint32_t first_bits = (uint32_t)bits[0] << 16;
        float first;
        memcpy(&first, &first_bits, sizeof(first));

        printf("%s %zu %" PRIu64 " %" PRIu64 " %.6g\n", t->name, t->numel, sum, weighted, first);
        free(bits);
    }
}
