# Glitch LLM

TinyLlama 1.1B (WebLLM) and Stable Diffusion 1.5 (Web Stable Diffusion) in the browser, with Vinesauce ROM Corruptor bit manipulation that corrupts the weights live in GPU memory.

```bash
bun install
bun run dev
```

Requires a browser with WebGPU and the `shader-f16` feature, basically modern chromium browsers only. Each model downloads once into OPFS: TinyLlama is about 630 MB, Stable Diffusion about 2 GB. Only one model sits on the GPU at a time.

- `src/gpu/corrupt.wgsl`, `src/gpu/corruptor.ts`: the corruption and undo shader.
- `src/stores/model.ts`: status, loading, crash restart and corruption, shared by both models.
- `src/models/tinyLlama.ts`, `src/models/stableDiffusion.ts`: one adapter per model.
- `public/websd/`: the Web Stable Diffusion runtime (Apache-2.0), vendored from websd.mlc.ai. 
