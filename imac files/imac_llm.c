/*
 * imac_llm.c — LLM inference on a 1998 iMac G3
 *
 * Based on Karpathy's llama2.c, adapted for:
 *   - PowerPC big-endian (pre-swapped model files)
 *   - Classic Mac OS 8.5 (no mmap, no terminal, no clock_gettime)
 *   - 32 MB RAM (static buffers, Mac Memory Manager)
 *   - Grouped-query attention (n_kv_heads != n_heads)
 *
 * Cross-compile with Retro68:
 *   cmake .. -DCMAKE_TOOLCHAIN_FILE=.../retroppc.toolchain.cmake
 *   make
 *
 * Usage on the iMac:
 *   1. Place imac_llm, stories260K_be.bin, tok512_be.bin in same folder
 *   2. Set memory to 3000 KB in Get Info
 *   3. Optionally create prompt.txt with a starting phrase
 *   4. Double-click imac_llm
 *   5. Open output.txt in SimpleText
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <Memory.h>

/* ---- Data structures ---- */

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int max_seq_len;
} Config;

typedef struct {
    float *token_embedding_table;
    float *rms_att_weight;
    float *rms_ffn_weight;
    float *wq;
    float *wk;
    float *wv;
    float *wo;
    float *w1;
    float *w2;
    float *w3;
    float *rms_final_weight;
    float *freq_cis_real;
    float *freq_cis_imag;
    float *wcls;
} TransformerWeights;

typedef struct {
    char **vocab;
    float *vocab_scores;
    int vocab_size;
    int max_token_length;
} Tokenizer;

/* ---- Static buffers for inference state ----
 * Sized for the 260K model: dim=64, hidden=172, layers=5,
 * heads=8, kv_heads=4, vocab=512, max_seq_len capped to 32.
 * Using static arrays avoids malloc failures on 32 MB RAM. */

static float s_x[64], s_xb[64], s_xb2[64];
static float s_hb[172], s_hb2[172];
static float s_q[64], s_k[64], s_v[64];
static float s_att[8 * 32];        /* n_heads * max_seq_len */
static float s_logits[512];         /* vocab_size */
static float s_key_cache[5 * 32 * 64];   /* n_layers * max_seq_len * kv_dim */
static float s_value_cache[5 * 32 * 64];

/* ---- Math operations ---- */

void rmsnorm(float *o, float *x, float *weight, int size) {
    float ss = 0.0f;
    int j;
    for (j = 0; j < size; j++)
        ss += x[j] * x[j];
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / (float)sqrt((double)ss);
    for (j = 0; j < size; j++)
        o[j] = weight[j] * (ss * x[j]);
}

void softmax(float *x, int size) {
    float max_val = x[0];
    float sum = 0.0f;
    int i;
    for (i = 1; i < size; i++)
        if (x[i] > max_val) max_val = x[i];
    for (i = 0; i < size; i++) {
        x[i] = (float)exp((double)(x[i] - max_val));
        sum += x[i];
    }
    for (i = 0; i < size; i++)
        x[i] /= sum;
}

void matmul(float *xout, float *x, float *w, int n, int d) {
    int i, j;
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (j = 0; j < n; j++)
            val += w[i * n + j] * x[j];
        xout[i] = val;
    }
}

/* ---- Weight initialization ----
 * IMPORTANT: Uses n_kv_heads * head_size for wk/wv sizing,
 * not dim * dim. The 260K model uses grouped-query attention
 * (kv_heads=4, heads=8) so wk and wv are smaller matrices. */

void checkpoint_init_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    w->token_embedding_table = ptr; ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;        ptr += p->n_layers * p->dim;
    w->wq = ptr;                    ptr += p->n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;                    ptr += p->n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;                    ptr += p->n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;                    ptr += p->n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;        ptr += p->n_layers * p->dim;
    w->w1 = ptr;                    ptr += p->n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;                    ptr += p->n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;                    ptr += p->n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;      ptr += p->dim;
    w->freq_cis_real = ptr;         ptr += p->max_seq_len * head_size / 2;
    w->freq_cis_imag = ptr;         ptr += p->max_seq_len * head_size / 2;
    if (shared_weights) {
        w->wcls = w->token_embedding_table;
    } else {
        w->wcls = ptr;
    }
}

/* ---- Transformer forward pass ---- */

float *forward(Config *p, TransformerWeights *w, int token, int pos) {
    float *content_row = w->token_embedding_table + token * p->dim;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    int l, i, h, t;
    float score, a;

    memcpy(s_x, content_row, dim * sizeof(float));

    for (l = 0; l < p->n_layers; l++) {
        /* Attention pre-norm */
        rmsnorm(s_xb, s_x, w->rms_att_weight + l * dim, dim);

        /* QKV projections */
        matmul(s_q, s_xb, w->wq + l * dim * dim, dim, dim);
        matmul(s_k, s_xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s_v, s_xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        /* RoPE positional encoding */
        for (i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = w->freq_cis_real[pos * head_size / 2 + head_dim / 2];
            float fci = w->freq_cis_imag[pos * head_size / 2 + head_dim / 2];
            float v0 = s_q[i];
            float v1 = s_q[i + 1];
            s_q[i] = v0 * freq - v1 * fci;
            s_q[i + 1] = v0 * fci + v1 * freq;
            if (i < kv_dim) {
                v0 = s_k[i];
                v1 = s_k[i + 1];
                s_k[i] = v0 * freq - v1 * fci;
                s_k[i + 1] = v0 * fci + v1 * freq;
            }
        }

        /* KV cache */
        {
            int loff = l * p->max_seq_len * kv_dim;
            float *key_cache_row = s_key_cache + loff + pos * kv_dim;
            float *value_cache_row = s_value_cache + loff + pos * kv_dim;
            memcpy(key_cache_row, s_k, kv_dim * sizeof(float));
            memcpy(value_cache_row, s_v, kv_dim * sizeof(float));
        }

        /* Multi-head attention */
        for (h = 0; h < p->n_heads; h++) {
            float *q_head = s_q + h * head_size;
            float *att_head = s_att + h * p->max_seq_len;
            int loff = l * p->max_seq_len * kv_dim;

            for (t = 0; t <= pos; t++) {
                float *k_head = s_key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                score = 0.0f;
                for (i = 0; i < head_size; i++)
                    score += q_head[i] * k_head[i];
                score /= (float)sqrt((double)head_size);
                att_head[t] = score;
            }
            softmax(att_head, pos + 1);

            {
                float *xb_head = s_xb + h * head_size;
                memset(xb_head, 0, head_size * sizeof(float));
                for (t = 0; t <= pos; t++) {
                    float *v_head = s_value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                    a = att_head[t];
                    for (i = 0; i < head_size; i++)
                        xb_head[i] += a * v_head[i];
                }
            }
        }

        /* Output projection + residual */
        matmul(s_xb2, s_xb, w->wo + l * dim * dim, dim, dim);
        for (i = 0; i < dim; i++)
            s_x[i] += s_xb2[i];

        /* FFN pre-norm */
        rmsnorm(s_xb, s_x, w->rms_ffn_weight + l * dim, dim);

        /* FFN with SwiGLU */
        matmul(s_hb, s_xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s_hb2, s_xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);
        for (i = 0; i < hidden_dim; i++) {
            float val = s_hb[i];
            val *= (1.0f / (1.0f + (float)exp(-(double)val)));
            val *= s_hb2[i];
            s_hb[i] = val;
        }
        matmul(s_xb, s_hb, w->w2 + l * hidden_dim * dim, hidden_dim, dim);
        for (i = 0; i < dim; i++)
            s_x[i] += s_xb[i];
    }

    /* Final norm + classifier */
    rmsnorm(s_x, s_x, w->rms_final_weight, dim);
    matmul(s_logits, s_x, w->wcls, dim, p->vocab_size);
    return s_logits;
}

/* ---- Tokenizer ---- */

int load_tokenizer(Tokenizer *t, const char *path, int vocab_size) {
    FILE *file = fopen(path, "rb");
    int i, len;
    if (!file) return 0;

    t->vocab_size = vocab_size;
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
    if (!t->vocab || !t->vocab_scores) { fclose(file); return 0; }

    fread(&t->max_token_length, sizeof(int), 1, file);
    for (i = 0; i < vocab_size; i++) {
        fread(&t->vocab_scores[i], sizeof(float), 1, file);
        fread(&len, sizeof(int), 1, file);
        t->vocab[i] = (char *)malloc(len + 1);
        if (!t->vocab[i]) { fclose(file); return 0; }
        fread(t->vocab[i], len, 1, file);
        t->vocab[i][len] = '\0';
    }
    fclose(file);
    return 1;
}

int str_lookup(const char *str, Tokenizer *t) {
    int i;
    for (i = 0; i < t->vocab_size; i++) {
        if (strcmp(str, t->vocab[i]) == 0) return i;
    }
    return -1;
}

/* BPE encoding: turns a text string into a sequence of token IDs */
int bpe_encode(const char *text, Tokenizer *t, int *tokens, int max_tokens) {
    int n_tokens = 0;
    int i, best_id, best_idx;
    float best_score;
    char merge[128];
    const char *p = text;

    /* First pass: encode each character as a single token */
    while (*p != '\0' && n_tokens < max_tokens) {
        char single[2];
        int id;
        single[0] = *p;
        single[1] = '\0';
        id = str_lookup(single, t);
        if (id == -1) { p++; continue; }
        tokens[n_tokens++] = id;
        p++;
    }

    /* Merge pass: greedily merge adjacent tokens */
    while (1) {
        best_score = -1e10f;
        best_id = -1;
        best_idx = -1;
        for (i = 0; i < n_tokens - 1; i++) {
            int len1 = strlen(t->vocab[tokens[i]]);
            int len2 = strlen(t->vocab[tokens[i+1]]);
            if (len1 + len2 < 127) {
                int id;
                strcpy(merge, t->vocab[tokens[i]]);
                strcat(merge, t->vocab[tokens[i+1]]);
                id = str_lookup(merge, t);
                if (id != -1 && t->vocab_scores[id] > best_score) {
                    best_score = t->vocab_scores[id];
                    best_id = id;
                    best_idx = i;
                }
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        {
            int j;
            for (j = best_idx + 1; j < n_tokens - 1; j++)
                tokens[j] = tokens[j+1];
        }
        n_tokens--;
    }

    return n_tokens;
}

/* ---- Sampling ---- */

int argmax(float *v, int n) {
    int max_i = 0;
    float max_p = v[0];
    int i;
    for (i = 1; i < n; i++) {
        if (v[i] > max_p) { max_p = v[i]; max_i = i; }
    }
    return max_i;
}

int sample(float *probabilities, int n) {
    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    int i;
    for (i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (r < cdf) return i;
    }
    return n - 1;
}

/* ---- File opener with path fallbacks ----
 * Classic Mac OS doesn't reliably set the working directory
 * to the app's folder. Try multiple path formats. */

FILE *try_open(const char *name, const char *mode) {
    FILE *f;
    char path[512];

    f = fopen(name, mode);
    if (f) return f;

    sprintf(path, ":untitled folder 1:%s", name);
    f = fopen(path, mode);
    if (f) return f;

    sprintf(path, "Macintosh HD:untitled folder 1:%s", name);
    f = fopen(path, mode);
    if (f) return f;

    return NULL;
}

/* ---- Main ---- */

int main() {
    Config config;
    Config orig_config;
    TransformerWeights weights;
    FILE *model_file;
    FILE *out;
    FILE *prompt_file;
    float *data;
    long file_size;
    int steps = 32;
    float temperature = 0.9f;
    int token, next, pos;
    float *logits;
    clock_t start, end_time;
    double elapsed;
    long weights_size;
    Tokenizer tokenizer;
    int shared_weights;
    char prompt[256];
    int prompt_tokens[64];
    int n_prompt_tokens;
    int i;

    /* Expand heap to maximum */
    MaxApplZone();
    MoreMasters();
    MoreMasters();
    MoreMasters();
    MoreMasters();

    /* Zero static buffers */
    memset(s_key_cache, 0, sizeof(s_key_cache));
    memset(s_value_cache, 0, sizeof(s_value_cache));
    memset(s_att, 0, sizeof(s_att));
    memset(s_logits, 0, sizeof(s_logits));

    /* Read prompt if available */
    prompt[0] = '\0';
    prompt_file = try_open("prompt.txt", "r");
    if (prompt_file) {
        fgets(prompt, 255, prompt_file);
        fclose(prompt_file);
        i = strlen(prompt);
        while (i > 0 && (prompt[i-1] == '\n' || prompt[i-1] == '\r'))
            prompt[--i] = '\0';
    }

    /* Open output file */
    out = fopen("output.txt", "w");
    if (!out) return 1;

    fprintf(out, "=== iMac G3 LLM ===\n");
    fprintf(out, "233 MHz PowerPC 750 | 32 MB RAM | Mac OS 8.5\n\n");
    fflush(out);

    /* Load model */
    model_file = try_open("stories260K_be.bin", "rb");
    if (!model_file) {
        fprintf(out, "ERROR: Cannot open stories260K_be.bin\n");
        fclose(out);
        return 1;
    }

    fread(&config, sizeof(Config), 1, model_file);
    orig_config = config;
    config.max_seq_len = 32;

    shared_weights = (config.vocab_size > 0) ? 1 : 0;
    if (config.vocab_size < 0) config.vocab_size = -config.vocab_size;
    if (orig_config.vocab_size < 0) orig_config.vocab_size = -orig_config.vocab_size;

    fprintf(out, "Model: dim=%d, layers=%d, heads=%d, vocab=%d\n",
            config.dim, config.n_layers, config.n_heads, config.vocab_size);

    /* Read weights */
    fseek(model_file, 0, SEEK_END);
    file_size = ftell(model_file);
    fseek(model_file, sizeof(Config), SEEK_SET);

    weights_size = file_size - sizeof(Config);
    data = (float *)NewPtr(weights_size);
    if (!data) data = (float *)malloc(weights_size);
    if (!data) {
        fprintf(out, "ERROR: Cannot allocate %ld bytes for weights\n", weights_size);
        fclose(out);
        fclose(model_file);
        return 1;
    }
    fread(data, 1, weights_size, model_file);
    fclose(model_file);

    checkpoint_init_weights(&weights, &orig_config, data, shared_weights);

    /* Load tokenizer */
    if (!load_tokenizer(&tokenizer, "tok512_be.bin", config.vocab_size)) {
        fprintf(out, "ERROR: Cannot load tok512_be.bin\n");
        fclose(out);
        return 1;
    }

    /* Encode prompt */
    if (prompt[0] != '\0') {
        n_prompt_tokens = bpe_encode(prompt, &tokenizer, prompt_tokens, 64);
        fprintf(out, "Prompt: \"%s\" (%d tokens)\n", prompt, n_prompt_tokens);
    } else {
        n_prompt_tokens = 0;
        fprintf(out, "No prompt.txt found, generating from BOS.\n");
    }

    fprintf(out, "Generating %d tokens...\n\n", steps);
    fflush(out);

    /* Generate */
    srand((unsigned int)time(NULL));
    token = 1; /* BOS */
    start = clock();

    for (pos = 0; pos < steps; pos++) {
        logits = forward(&config, &weights, token, pos);

        if (pos < n_prompt_tokens) {
            /* Force prompt tokens */
            next = prompt_tokens[pos];
        } else {
            /* Sample from distribution */
            int q;
            for (q = 0; q < config.vocab_size; q++)
                logits[q] /= temperature;
            softmax(logits, config.vocab_size);
            next = sample(logits, config.vocab_size);
        }

        fprintf(out, "%s", tokenizer.vocab[next]);
        fflush(out);
        token = next;
    }

    end_time = clock();
    elapsed = (double)(end_time - start) / CLOCKS_PER_SEC;

    fprintf(out, "\n\n--- STATS ---\n");
    fprintf(out, "Tokens: %d\n", steps);
    fprintf(out, "Time: %.2f seconds\n", elapsed);
    if (elapsed > 0)
        fprintf(out, "Speed: %.2f tok/s\n", (double)steps / elapsed);
    fprintf(out, "-------------\n");
    fclose(out);

    return 0;
}
