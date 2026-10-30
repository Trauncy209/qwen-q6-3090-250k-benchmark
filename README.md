# Single RTX 3090: Swift 1.5 Qwen3.8 27B Q6_K, native MTP, 250K populated context

## What was actually demonstrated

An unchanged, publicly downloadable Q6_K GGUF ran on one RTX 3090, with 262,144 allocated context and successful uncached inference at 250,000 populated prompt tokens. Memory fit comes from the runtime's compressed KV/state and draft context, not converting the Q6 weights to a smaller weight quant. We did not invent SJ-KVaRN or MTP: credit belongs to the model authors, quantizer and llamAmpere/llama.cpp contributors.

This is a useful fit/speed/retrieval result, **not a claim of being first, unprecedented, lossless KV compression, or superior coding intelligence**. The complete homelab agent system uses additional GPUs and some cloud helpers; only the main Q6 inference benchmark is single-3090.

## Hardware and software

- Main machine: AMD Ryzen 9 5950X, 16 cores / 32 threads; nominal 64GB system RAM (65,709,868 KiB reported).
- GPU: NVIDIA RTX 3090, 24,576 MiB; 350W configured power limit, **not a measured benchmark power draw**.
- Host: Fedora Linux 44 KDE; kernel 7.2.11-200.fc44.x86_64; NVIDIA driver 615.78.08.
- Runtime container: Ubuntu 24.04-derived local image, CUDA 13.1.1 (cudart package 13.1.80-1). The local hardened image is not a published download.
- Backend: [JakeATX/llamAmpere](https://github.com/JakeATX/llamAmpere), v0.6 source commit [`acbeece8851bc36a3a2437ad6094fb7ce1a635f8`](https://github.com/JakeATX/llamAmpere/tree/acbeece8851bc36a3a2437ad6094fb7ce1a635f8).
- Build: Release; CUDA enabled; compute capability 86; native CPU optimizations; CUDA graphs and flash attention enabled. The recorded build also enabled NCCL, although this benchmark does not use distributed inference.
- Local CUDA build guard: `ampere-sm86.patch` is included. It prevents instantiating oversized eight-warp row-kernel shapes exceeding 48KiB static shared memory. It does not alter model weights. Applying this patch is not by itself a fresh build certification.
- Template: [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates), v22.5. Included `chat_template.jinja` is byte-identical to the public file checked during packaging.

## Exact target weights and integrity

- Filename: `Swift-1.5-Qwen3.8-27B-Uncensored-BF16.i1-Q6_K.gguf`
- Quant: imatrix Q6_K, **not BF16**; “BF16” is part of the source model name.
- Size: **22,431,001,472 bytes**.
- Local full-file SHA256, independently matched to public Hugging Face LFS metadata:
  `20fe64906cdf91a0e83db2dad3d1d73c0f56829410dbc54eb76e74f1b29839e8`
- GGUF version 3, 866 tensors, architecture identifier `qwen35`, declared context 262144. The architecture identifier and the product/model name need not be identical.
- The file contains usable native MTP/NextN components; no separate draft-model weight download was needed.
- [Quant repository](https://huggingface.co/mradermacher/Swift-1.5-Qwen3.8-27B-Uncensored-BF16-i1-GGUF)
- [Pinned direct model download](https://huggingface.co/mradermacher/Swift-1.5-Qwen3.8-27B-Uncensored-BF16-i1-GGUF/resolve/2c5a4bf01d3c9c16170355574bb04535caa8ba28/Swift-1.5-Qwen3.8-27B-Uncensored-BF16.i1-Q6_K.gguf)
- [Upstream unquantized model](https://huggingface.co/d0xin/Swift-1.5-Qwen3.8-27B-Uncensored-BF16)
- [Swift family](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b)

## Exact tested serving settings

```bash
./llama-server \
  -m ./Swift-1.5-Qwen3.8-27B-Uncensored-BF16.i1-Q6_K.gguf \
  --alias qwen-atx-27b \
  -c 262144 -ngl 99 -fa on -fit off \
  -ctk sjkvarn3 -ctv sjkvarn2 -cts q8_0 \
  --sjkvarn-body-type auto \
  -b 256 -ub 64 --parallel 1 \
  --spec-type draft-mtp-adaptive --spec-draft-sjkvarn \
  --no-spec-mb-mtp \
  --spec-draft-n-min-adaptive 3 --spec-draft-n-max 4 --spec-draft-p-min 0 \
  --reasoning-budget 512 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
  --cache-ram 0 --cache-disk-limit 0 \
  --jinja --chat-template-file ./chat_template.jinja \
  --host 127.0.0.1 --port 8105
```

The benchmark request overrides temperature to zero and disables thinking via the rendered template. `--no-spec-mb-mtp` selects the runtime's exact/non-mode-biased target-verification path. That is **not** a statement that quantized KV/state is lossless relative to uncompressed target execution. No vision projector is loaded in the main Q6 runtime. Layers were requested fully GPU-offloaded with `-ngl 99`, fit disabled; no RPC, tensor split, unified-memory trick or deliberate CPU layer offload was used. The reported VRAM is whole-card sampled usage, not just the model buffer.

SJ-KVaRN and compressed draft support are part of the linked fork: there is no separate KV-model file to download. See the [pinned v0.6 release notes](https://github.com/JakeATX/llamAmpere/blob/acbeece8851bc36a3a2437ad6094fb7ce1a635f8/docs/llamampere-v0.6/RELEASE_NOTES.md) and the fork's CLI/source for the precise implementation.

## Measurements

| Populated prompt | Q6 + exact MTP generation | Q6 prefill | Q6 prefill time | Peak sampled VRAM | Retrieval |
|---:|---:|---:|---:|---:|---:|
|4,096|69.76 tok/s|699.9 tok/s|See raw records|See raw records|Not scored in this first fit test|
|32,768|63.56 tok/s|581.7 tok/s|See raw records|24,158 MiB reported during earlier fit measurements|Not scored in this first fit test|
|131,072|44.13 tok/s|380.33 tok/s|344.62 s|24,108 MiB|4/4|
|250,000|34.17 tok/s|263.53 tok/s|948.64 s|24,124 MiB|4/4|

The first two figures are means of two trials from the earlier fit experiment. Deep-context figures are **one trial each**, with 128 generated tokens, zero prompt-cache hits, full prompt evaluation and no truncation. Draft acceptance was 89/115 (77.39%) on each deep JSON fixture; acceptance is workload-dependent, not a universal property.

### Q5 comparison—not a pure weight-quant experiment

The preserved Little Qwen uses **SJ-KVaRN 3-bit keys / 3-bit values**, unlike the Q6 compromise's **3-bit keys / 2-bit values**. Q5 also uses batch1024/ubatch256 rather than Q6 batch256/ubatch64. Token vocabulary, token IDs, token types and merges matched across both GGUFs.

| Populated prompt | Earlier Q5 generation | Q6 generation | Earlier Q5 prefill | Q6 prefill |
|---:|---:|---:|---:|---:|
|131,072|50.45 tok/s|44.13 tok/s|651.87 tok/s|380.33 tok/s|
|250,000|32.57 tok/s|34.17 tok/s|472.37 tok/s|263.53 tok/s|

Q6 decode is comparable at 250K in this fixture, but **its cold prefill is slower**. Do not treat a roughly 5% single-trial decode difference as a reliable win. This is not 64 tok/s with 250K filled; 63.56 tok/s was measured at 32K.

## Methodology and honest limits

The reproducible prompt consists of deterministic archive records and four authoritative REGISTER lines at approximately 12%, 43%, 74% and 96% of filler depth. Their values must be returned as JSON. These four checks demonstrate retrieval at those positions, not perfect access to every token or arbitrary long-context reasoning.

`reproduce.py` generates the same four prompt-token fixtures as the original experiment; its output hashes were verified identical against all four saved originals using the deployed Q6 tokenizer/template. It measures server prefill/decode timings and wall-clock time to first content. Its speed runs deliberately set `ignore_eos=true` for a fixed 128-token output. Consequently, the model repeats JSON after its first answer; those forced continuations are **not evidence of spontaneous agent loops**, and the rates should not be advertised as representative coding throughput. No coding benchmark, broad quality suite or failure-recovery study was completed by this deep-context test.

To prepare or rerun against your own local endpoint:

```bash
python3 reproduce.py --port 8105 --prepare-only
python3 reproduce.py --port 8105 --sizes 131072 250000
```

This can monopolize the model for many minutes and uses hundreds of thousands of uncached tokens. Benchmark raw measurements are included. The public reproduction script and fixture generation were exercised; a clean-machine CUDA build and another complete deep benchmark from the public bundle were not rerun.

For building, use the pinned fork's build instructions, CUDA13.1 and compute capability86; apply the included patch if reproducing this exact checkout. The corresponding CMake settings are recorded in `hardware-build.json`. Do not assume upstream vanilla llama.cpp accepts the fork-specific cache or MTP switches.

## Actual app integration and supporting machines

- **Big Qwen:** the saved Q6/3-bit-key/2-bit-value native-MTP recipe on the single3090. App route is the existing separate Pi harness, retaining sessions and 262144 context / 131072 configured maximum output. The existing approximately132K compaction threshold was retained, not advertised as250K resident conversation history.
- **Little Qwen:** preserved Q5 weights/launcher and independent Hermes harness. Little and Big time-share the3090; they do not reside there simultaneously.
- **Subagents:** shared Spark X2.6 4B Q8_0 weights and Q8_0 KV on a separate Node2, RTX4060Ti16GB plus RTX5060Ti16GB. Ten32768-context slots per GPU, twenty total;999-turn configured task budget. Big's real app test invoked `delegate_task(action=list)` through the same owner-scoped broker. This verifies integration, not every worker's task reliability. One inventory helper failed to deliver a final report during preparation; the parent independently gathered the evidence instead.
- **Vision upgrade:** Qwen3.5-4B Q4_K_M on Node2's4060Ti, **alongside unchanged workers**; F16 projector computed on CPU to save VRAM;8192 context, one vision slot, Q8 KV. The old0.8B was disabled, not deleted. A real concurrent test returned the correct red square and number42 in5.01s while both worker endpoints returned WORKER_READY. Minimum sampled available Node2 RAM8.49GiB;4060Ti sampled15213MiB used. This is a small concurrency check, **not twenty busy slots plus vision stress certification**. The shared transport retains a legacy0.8B API alias for compatibility; actual `/props` points to the4B GGUF.
- **Image generation:** Qwen Image2.1 Turbo INT8 ConvRot, W4A8 Qwen3-VL8B text encoder, BF16 VAE, eight Euler steps/CFG1 on the5060Ti, borrowing it on demand through the existing exclusive lease. Big's provider produced a real1024-square PNG; the primary3090 text runtime was not used for generation.
- **Auxiliary services:** Big's configuration matches Little's helper routing. Some title/compression/approval/etc. helpers use GPT through the existing private proxy; compact background-review configuration points to Spark. The Pi harness does not necessarily exercise every configured auxiliary section. **This complete app is not all-local**, and these helpers did not participate in the standalone main-model benchmark.

## Where to get every core component

### Required for the headline single3090 test

1. [Pinned Q6 GGUF download](https://huggingface.co/mradermacher/Swift-1.5-Qwen3.8-27B-Uncensored-BF16-i1-GGUF/resolve/2c5a4bf01d3c9c16170355574bb04535caa8ba28/Swift-1.5-Qwen3.8-27B-Uncensored-BF16.i1-Q6_K.gguf).
2. [Pinned llamAmpere source](https://github.com/JakeATX/llamAmpere/tree/acbeece8851bc36a3a2437ad6094fb7ce1a635f8), [releases](https://github.com/JakeATX/llamAmpere/releases), [upstream llama.cpp](https://github.com/ggml-org/llama.cpp).
3. [Template repository](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates), [direct template](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/chat_template.jinja); exact tested copy included.
4. [CUDA13.1 archive](https://developer.nvidia.com/cuda-13-1-0-download-archive), [NVIDIA CUDA containers](https://hub.docker.com/r/nvidia/cuda), [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), [Docker Engine](https://docs.docker.com/engine/install/), [NCCL](https://developer.nvidia.com/nccl), [CMake](https://cmake.org/download/), [Git](https://git-scm.com/downloads), [Python](https://www.python.org/downloads/), [Fedora](https://fedoraproject.org/).
5. Benchmark scripts, source patch, exact template and raw evidence: included in this bundle. The owner's private hardened container image/app is not publicly distributed; there is no public link to pretend otherwise.

### Optional full-agent supporting stack—not needed for singleGPU benchmark

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) and [documentation](https://hermes-agent.nousresearch.com/docs).
- [Pi coding-agent](https://github.com/earendil-works/pi), package `@earendil-works/pi-coding-agent` v1.4.0; [Node.js](https://nodejs.org/en/download).
- [Spark original](https://huggingface.co/XHToken/Spark-X2.6-4B), [Q8 GGUF download source](https://huggingface.co/bartowski/Spark-X2.6-4B-GGUF/blob/main/Spark-X2.6-4B-Q8_0.gguf). This is a verified public distribution of the same model/quant, not a claimed hash match to the renamed installed worker file.
- [Qwen3.5-4B original](https://huggingface.co/Qwen/Qwen3.5-4B), [installed quant repository](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF), [pinned Q4_K_M weights](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf), [pinned F16 projector](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/mmproj-F16.gguf). Both installed files were SHA256-verified against public metadata.
- [Old0.8B quant/projector distribution](https://huggingface.co/prithivMLmods/Qwen3.5-0.8B-f32-GGUF), retained only for rollback.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI), [Qwen Image2.1 Turbo ConvRot quant](https://huggingface.co/NidAll/Turbo-INT8-ConvRot-for-Qwen-Image-2.1), [alternative upstream quant distribution](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-FP8), [image text encoder](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/text_encoders/qwen3vl_8b_w4a8.safetensors), [image VAE](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/vae/qwen_image_2.1_vae_bf16.safetensors).
- Optional cloud helpers require the owner's separately authenticated provider setup; [OpenAI developer documentation](https://platform.openai.com/docs). No credentials, proxy tokens, owner conversations or private service configuration are included.

Read each model's license and upstream documentation. These links identify available components; they do not imply endorsement, identical performance on another system, or permission to redistribute model weights inside this evidence pack.

## Publication privacy and licensing

Only synthetic benchmark prompts/results and selected technical provenance are included. Host paths, private network addresses, credentials, personal names/emails, chat histories, deployment scripts and original Git history are excluded. Wall-clock timestamps and detailed slot telemetry were removed from public measurement exports; inference timings and results were preserved.

The included template is from froggeric/Qwen-Fixed-Chat-Templates, whose model-card license is Apache-2.0. Its license text is included in THIRD-PARTY-APACHE-2.0.txt; see NOTICE.md for scope and attribution. Model weights are linked, not redistributed.
