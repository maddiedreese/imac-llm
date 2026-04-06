# LLM on a 1998 iMac G3 (32 MB RAM)

Running a language model locally on a stock 1998 iMac G3 with 32 MB of RAM, Mac OS 8.5, and a 233 MHz PowerPC 750 processor. No hardware upgrades.

**Prompt:** "The green goblin"  
**Output:** "The green goblin had a big mop. She had a cow in the field too. I"

![iMac G3 output](screenshots/output.jpg)

## What is this?

This is a port of [Karpathy's llama2.c](https://github.com/karpathy/llama2.c) to classic Mac OS, targeting the original Bondi Blue iMac G3. It runs the 260K parameter TinyStories model (Llama 2 architecture) with a ~1 MB checkpoint entirely in local memory.

The iMac has 32 MB of RAM — about 500x less than a modern laptop. The model generates coherent children's stories from a text prompt.

## How it works

1. **You type a prompt** into `prompt.txt` using SimpleText on the iMac
2. **The app tokenizes it** using BPE encoding with a 512-token vocabulary
3. **Runs transformer inference** — matrix multiplies, RoPE, attention, SwiGLU, the whole forward pass
4. **Writes the continuation** to `output.txt`, which you open in SimpleText

32 tokens generate in under a second on the 233 MHz G3.

## Hardware

| Spec | Value |
|------|-------|
| Model | iMac G3 Rev B (October 1998) |
| CPU | 233 MHz PowerPC 750 (G3) |
| RAM | 32 MB (stock, no upgrades) |
| OS | Mac OS 8.5 |
| Display | 15" CRT |
| Storage | 4 GB HDD |

## The Model

[Karpathy's TinyStories 260K](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K) — a 260,000 parameter model with the Llama 2 architecture, trained on children's stories.

| Parameter | Value |
|-----------|-------|
| dim | 64 |
| hidden_dim | 172 |
| n_layers | 5 |
| n_heads | 8 |
| n_kv_heads | 4 |
| vocab_size | 512 |
| max_seq_len | 512 (capped to 32 at runtime) |
| Checkpoint size | ~1 MB |

## Building

### Prerequisites

- A modern Mac or Linux machine for cross-compilation
- [Retro68](https://github.com/autc04/Retro68) — a GCC-based cross-compiler for classic Mac OS
- Python 3 (for endian swapping)
- An iMac G3 (or other PowerPC classic Mac OS machine) on the same network

### Step 1: Build Retro68

```bash
# Install dependencies (macOS)
brew install gcc cmake gmp mpfr libmpc bison texinfo boost

# Clone and build (PPC only)
git clone --recursive https://github.com/autc04/Retro68.git
cd ..
mkdir Retro68-build && cd Retro68-build
../Retro68/build-toolchain.bash --no-68k
```

This takes 30-60 minutes. It builds a full GCC cross-compiler targeting PowerPC classic Mac OS.

### Step 2: Download and endian-swap the model

The iMac's PowerPC CPU is big-endian, but the model files are little-endian. Every 32-bit value needs to be byte-swapped.

```bash
mkdir imac-llm && cd imac-llm

# Download model and tokenizer
curl -L -o stories260K.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin
curl -L -o tok512.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.bin

# Endian swap for PowerPC
python3 endian_swap.py
```

This produces `stories260K_be.bin` and `tok512_be.bin`.

### Step 3: Build the app

```bash
export PATH=/path/to/Retro68-build/toolchain/bin:$PATH

mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=/path/to/Retro68-build/toolchain/powerpc-apple-macos/cmake/retroppc.toolchain.cmake
make
```

This produces `imac_llm.bin` (MacBinary), `imac_llm.dsk` (floppy image), and `imac_llm.APPL`.

### Step 4: Transfer to the iMac

Serve the files over FTP from your modern machine:

```bash
pip3 install pyftpdlib
python3 -m pyftpdlib -p 2121 -w -u mac -P mac
```

On the iMac, open a web browser and navigate to `ftp://mac:mac@YOUR_IP:2121/`. Download:
- `imac_llm.bin`
- `stories260K_be.bin`
- `tok512_be.bin`

### Step 5: Run

1. Place all three files in the same folder on the iMac
2. Select `imac_llm`, do File → Get Info, switch to Memory, set Preferred Size to **3000**
3. Create `prompt.txt` in the same folder with your starting phrase
4. Double-click `imac_llm`
5. Open `output.txt` to read the generated text

## Technical Challenges

### Endianness

The PowerPC 750 is big-endian. All model checkpoints and tokenizer files from llama2.c are little-endian. The `endian_swap.py` script converts every 32-bit int and float in the model checkpoint, and the int/float fields in the tokenizer (but not the raw byte strings).

### Memory management

Mac OS 8.5 gives each application a fixed memory partition, defaulting to well under 1 MB. The model weights alone are ~1 MB. Solutions:

- **`MaxApplZone()`** — expands the app's heap to its maximum allowed size
- **`NewPtr()`** — allocates from the Mac Memory Manager directly instead of `malloc`
- **Static buffers** — all inference state (KV cache, attention, activations) uses compile-time static arrays instead of dynamic allocation
- **Capped `max_seq_len`** — reduced from 512 to 32 to shrink the KV cache by 16x
- **Set memory partition in Get Info** — the user must manually increase the app's memory allocation to 3000 KB

### RetroConsole doesn't work

Retro68's `RetroConsole` library (which provides `printf` via a Mac window) crashes on the iMac G3 Rev B. All output is written to `output.txt` instead, which you open in SimpleText.

### Grouped-Query Attention weight layout

This was the hardest bug. The 260K model uses GQA with `n_kv_heads=4` and `n_heads=8`. The original llama2.c `checkpoint_init_weights` function calculates pointer offsets for `wk` and `wv` using `dim * dim`, which assumes `n_kv_heads == n_heads`. When they differ, `wk` and `wv` are smaller matrices, and every weight pointer after them (wo, rms_ffn_weight, w1, w2, w3, rms_final_weight, freq_cis) ends up pointing to the wrong memory location.

The fix:
```c
// Wrong (original llama2.c for this model)
w->wk = ptr; ptr += p->n_layers * p->dim * p->dim;
w->wv = ptr; ptr += p->n_layers * p->dim * p->dim;

// Correct
w->wk = ptr; ptr += p->n_layers * p->dim * (p->n_kv_heads * head_size);
w->wv = ptr; ptr += p->n_layers * p->dim * (p->n_kv_heads * head_size);
```

This manifested as `freq_cis_real` values of -1.87 trillion instead of ~1.0, which produced NaN through the entire forward pass.

### No SSH, no terminal

Mac OS 8.5 has no SSH, no terminal emulator, no command line. File transfer is via FTP or HTTP. Debugging is via writing diagnostic info to `output.txt` and checking it in SimpleText after each run.

## File overview

```
imac_llm.c          — The inference engine (single file, ~300 lines)
endian_swap.py       — Converts model/tokenizer to big-endian
CMakeLists.txt       — Build configuration for Retro68
imac_llm.r           — Mac resource file (SIZE resource for memory partition)
```

## Prior art

- [EXO Labs — Llama on Windows 98](https://blog.exolabs.net/day-4/) — 260K model on a Pentium II (350 MHz, **128 MB RAM**, Windows 98). The inspiration for this project.
- [Resistor Network — LLMs on PowerPC](http://www.theresistornetwork.com/2025/03/thinking-different-thinking-slowly-llms.html) — 110M TinyStories on a PowerBook G4 (1.5 GHz, 1 GB RAM, Mac OS X).

This project runs on **4x less RAM** than the Pentium II project and targets a harder platform (PowerPC big-endian, classic Mac OS, no terminal).

## Credits

- [Andrej Karpathy](https://github.com/karpathy) — llama2.c and the TinyStories models
- [Wolfgang Thaller](https://github.com/autc04) — Retro68 cross-compiler
- [EXO Labs](https://github.com/exo-explore/llama98.c) — llama98.c, proving vintage hardware can run LLMs
- Built with help from [Claude](https://claude.ai) by Anthropic

## License

MIT
