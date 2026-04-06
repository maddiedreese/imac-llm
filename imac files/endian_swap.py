#!/usr/bin/env python3
"""
Endian-swap llama2.c model checkpoint and tokenizer for big-endian PowerPC.

The checkpoint format:
  - Config header: 7 x int32
  - Weights: all float32 (swapped as raw uint32)

The tokenizer format:
  - max_token_length: 1 x int32
  - For each token: score (float32), len (int32), bytes (raw, NOT swapped)

Usage:
  python3 endian_swap.py

Reads stories260K.bin and tok512.bin, produces stories260K_be.bin and tok512_be.bin.
"""
import struct

def swap_model(input_path, output_path):
    with open(input_path, 'rb') as f:
        data = f.read()

    n_values = len(data) // 4
    values = struct.unpack('<' + 'I' * n_values, data)
    swapped = struct.pack('>' + 'I' * n_values, *values)

    with open(output_path, 'wb') as f:
        f.write(swapped)

    config = struct.unpack('<7i', data[:28])
    labels = ['dim', 'hidden_dim', 'n_layers', 'n_heads', 'n_kv_heads', 'vocab_size', 'max_seq_len']
    print("Model config (%s):" % input_path)
    for label, val in zip(labels, config):
        print("  %s: %d" % (label, val))
    print("Total file size: %d bytes (%.1f KB)" % (len(data), len(data) / 1024.0))
    print("Swapped model written to: %s" % output_path)

def swap_tokenizer(input_path, output_path):
    with open(input_path, 'rb') as f:
        data = f.read()

    out = bytearray()
    pos = 0

    max_token_length = struct.unpack_from('<i', data, pos)[0]
    out += struct.pack('>i', max_token_length)
    pos += 4
    print("Tokenizer max_token_length: %d" % max_token_length)

    token_count = 0
    while pos < len(data):
        score = struct.unpack_from('<f', data, pos)[0]
        out += struct.pack('>f', score)
        pos += 4

        slen = struct.unpack_from('<i', data, pos)[0]
        out += struct.pack('>i', slen)
        pos += 4

        out += data[pos:pos + slen]
        pos += slen
        token_count += 1

    with open(output_path, 'wb') as f:
        f.write(bytes(out))

    print("Tokenizer tokens: %d" % token_count)
    print("Swapped tokenizer written to: %s" % output_path)

if __name__ == '__main__':
    swap_model('stories260K.bin', 'stories260K_be.bin')
    print()
    swap_tokenizer('tok512.bin', 'tok512_be.bin')
