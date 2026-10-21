# Glitch LLM

TinyLlama 1.1B in the browser through WebLLM, with Vinesauce ROM Corruptor bit manipulation that corrupts the weights live in GPU memory.

```bash
bun install
bun run dev
```

Requires a browser with WebGPU and the `shader-f16` feature, basically modern chromium browsers only. The model, about 630 MB, downloads once from Hugging Face into OPFS.

- `src/gpu/corrupt.wgsl`, `src/gpu/corruptor.ts`: the corruption and undo shader.
- `src/stores/llm.ts`: model loading, streaming, crash restart.
