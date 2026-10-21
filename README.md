# Glitch LLM

Generating ridiculous text synthesizing weird images with TinyLlama 1.1B and Stable Diffusion 1.5 through WebLLM in the browser. with Vinesauce ROM Corruptor bit manipulation as a WGSL shader that corrupts the weights live in GPU memory for laffs.

<img width="1728" height="962" alt="image" src="https://github.com/user-attachments/assets/87558c00-acd6-41d8-af1e-27942f1468e0" />

Requires a browser with WebGPU and the `shader-f16` feature, basically modern chromium browsers only. Each model downloads once into OPFS: TinyLlama is about 630 MB, Stable Diffusion about 2 GB. Only one model sits on the GPU at a time.


## How to use

1. Go to https://x8bitrain.github.io/glitch-llm/
2. Click on Load TinyLLama or Load Stable Diffusion (in image corruptions).
3. Wait for model to download from HF, the model gets stored in [OPFS](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system)
4. Tune your corruption settings, or use the default, and hit the CORRUPT button.
5. Enter your prompt and try to understand the output lol

## Why???

Because it's funny lol.

This idea was based on an experiment I did some years ago to see what would happen if I flipped bits in the weights of a model https://btr.pm/blog/giving-ai-brain-damage/
Since then the world of LLMs have significantly progressed and many of them can run in browsers. Also with wider webGPU adoption this becomes easier to poke around at the internals of a model running in memory. Now that browser tech has advanced enough for this to works I made this playground that implements the ideas in the blog post for anyone to screw around with.

## Developing

```bash
bun install
bun run dev
```

main stuff:

- `src/gpu/corrupt.wgsl`, `src/gpu/corruptor.ts`: the corruption and undo shader.
- `src/stores/model.ts`: status, loading, crash restart and corruption, shared by both models.
- `src/models/tinyLlama.ts`, `src/models/stableDiffusion.ts`: one adapter per model.
- `public/websd/`: the Web Stable Diffusion runtime (Apache-2.0) from websd.mlc.ai. 
