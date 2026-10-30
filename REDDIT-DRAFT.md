# Title

RTX 3090: Qwen 27B Q6_K + exact MTP at 250K populated context — 34 tok/s, with reproducible settings and caveats

# Post

I got Swift 1.5 Qwen3.8 27B's public Q6_K GGUF running on a single 24GB RTX 3090 with a 262,144-token context allocation, then actually populated 131K and 250K tokens rather than stopping at a successful startup.

The model weights stayed Q6_K. The fit comes from **llamAmpere v0.6's SJ-KVaRN 3-bit keys / 2-bit values, quantized recurrent state and compressed native-MTP draft cache**. Credit to JakeATX/llamAmpere, llama.cpp, the model authors, mradermacher and froggeric; these are their implementations/components, not a new cache algorithm I invented.

## Results

| Actually populated prompt | Generation | Prompt processing | Cold prefill time | Peak sampled whole-card VRAM | Retrieval |
|---:|---:|---:|---:|---:|---:|
|4,096|69.76 tok/s|699.9 tok/s|—|See raw records|Not scored in initial fit test|
|32,768|63.56 tok/s|581.7 tok/s|—|About24.2GB reported|Not scored in initial fit test|
|131,072|44.13 tok/s|380.33 tok/s|344.62s|24,108MiB|4/4|
|250,000|34.17 tok/s|263.53 tok/s|948.64s|24,124MiB|4/4|

Main host: Ryzen9 5950X, nominal64GB RAM, RTX3090; Fedora44, NVIDIA615.78.08, CUDA13.1.1 in an Ubuntu24.04-derived container. One inference slot; no RPC or deliberate CPU layer offload. The main-model benchmark has no vision projector loaded.

## Exact recipe

```bash
llama-server -m Swift-1.5-Qwen3.8-27B-Uncensored-BF16.i1-Q6_K.gguf \
  --alias qwen-atx-27b -c 262144 -ngl 99 -fa on -fit off \
  -ctk sjkvarn3 -ctv sjkvarn2 -cts q8_0 --sjkvarn-body-type auto \
  -b 256 -ub 64 --parallel 1 \
  --spec-type draft-mtp-adaptive --spec-draft-sjkvarn --no-spec-mb-mtp \
  --spec-draft-n-min-adaptive 3 --spec-draft-n-max 4 --spec-draft-p-min 0 \
  --reasoning-budget 512 --temp 1 --top-p 0.95 --top-k 20 --min-p 0 \
  --cache-ram 0 --cache-disk-limit 0 \
  --jinja --chat-template-file chat_template.jinja --host 127.0.0.1 --port 8105
```

The benchmark request uses temperature0 and thinking disabled. `--no-spec-mb-mtp` keeps the fork's exact/non-mode-biased speculative acceptance path. Draft acceptance was89/115, or77.39%, on these deep-context fixtures. That's workload-specific; it isn't a universal acceptance rate or proof that compressed KV itself is lossless.

## Download/source links

- [Exact Q6 GGUF, pinned revision](https://huggingface.co/mradermacher/Swift-1.5-Qwen3.8-27B-Uncensored-BF16-i1-GGUF/resolve/2c5a4bf01d3c9c16170355574bb04535caa8ba28/Swift-1.5-Qwen3.8-27B-Uncensored-BF16.i1-Q6_K.gguf)
- [Source model](https://huggingface.co/d0xin/Swift-1.5-Qwen3.8-27B-Uncensored-BF16)
- [Exact llamAmpere source revision](https://github.com/JakeATX/llamAmpere/tree/acbeece8851bc36a3a2437ad6094fb7ce1a635f8)
- [Fork release notes](https://github.com/JakeATX/llamAmpere/blob/acbeece8851bc36a3a2437ad6094fb7ce1a635f8/docs/llamampere-v0.6/RELEASE_NOTES.md)
- [Froggeric v22.5 template](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/chat_template.jinja)
- [CUDA containers](https://hub.docker.com/r/nvidia/cuda), [CUDA13.1](https://developer.nvidia.com/cuda-13-1-0-download-archive), [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Full-file Q6 SHA256, matched locally to the public download:

```text
20fe64906cdf91a0e83db2dad3d1d73c0f56829410dbc54eb76e74f1b29839e8
```

SJ-KVaRN and compressed draft are included in the fork, not separate downloads. The exact checkout has a small local CUDA shared-memory build guard; the source diff belongs with the reproduction pack.

## Don't skip these caveats

- These are a **small synthetic retrieval/speed benchmark**, not coding intelligence, perplexity or a general agent-quality result.
- We generated128 tokens with EOS deliberately ignored so output length was consistent. That produces repeated JSON after the initial answer and makes it a narrow decode workload. Don't interpret those repetitions as spontaneous agent loops—or assume the throughput represents normal coding.
- Four registry values were placed around12%,43%,74%,96% of filler depth. All were retrieved correctly at both deep sizes. This doesn't establish perfect access to every token.
- The deep points are one trial each. Short points are means of two earlier trials.
- **34 tok/s at250K, not64 tok/s at250K.** The63.56 figure belongs to32K depth.
- Against our earlier Q5 setup, decode at250K was32.57tok/s versus34.17tok/s here, but Q6 prefill was considerably slower:263.53 versus472.37tok/s. Q5 had3/3-bit KV and larger batching, so this isn't a clean weight-quant comparison or a reliable5% win.
- No claim this is the first such configuration. The interesting part is the specific unchangedQ6 fit, full-depth execution and reproducible measurements.

I also use this in a custom Hermes/Pi app. That wider setup uses additional GPUs for Spark workers, Qwen3.5-4B vision and Qwen Image2.1, plus some cloud auxiliary helpers. **Those are separate from the single3090 main-model result.** You do not need that agent stack to rerun the benchmark.

The reproducible scripts, raw synthetic measurements, exact template, local build patch and complete component links are available here: https://github.com/Trauncy209/qwen-q6-3090-250k-benchmark . This repository does not include model weights or private app credentials.
