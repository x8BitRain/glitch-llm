import type { ChatMessage } from '../stores/chats'
import { createOpfsCache, type OpfsCache } from './opfsCache'
import { readParamTensors, withTimeout, type ModelDefinition, type TvmArray, type TvmInstance } from './model'

// Port of mlc-ai/web-stable-diffusion's stable_diffusion.js (DPM-solver scheduler only), minus its DOM wiring.
const WEBSD_DIST = 'https://websd.mlc.ai/dist/'
const WASM_URL = `${WEBSD_DIST}stable_diffusion_webgpu.wasm`
const SCHEDULER_URL = `${WEBSD_DIST}scheduler_dpm_solver_multistep_consts.json`
const WEIGHTS_URL = 'https://huggingface.co/mlc-ai/web-sd/resolve/main/web-sd-shards-v1-5/'
const WEIGHTS_INDEX_URL = `${WEIGHTS_URL}ndarray-cache.json`
const TOKENIZER_URL = 'https://huggingface.co/openai/clip-vit-large-patch14/raw/main/tokenizer.json'
const VENDORED_DIR = `${import.meta.env.BASE_URL}websd/`
const RUNTIME_SCRIPTS = ['tvmjs_runtime.wasi.js', 'tvmjs.bundle.js']
const TOKENIZER_MODULE = 'tokenizers-wasm/tokenizers_wasm.js'

const MAX_TOKEN_LENGTH = 77
const LATENT_SHAPE = [1, 4, 64, 64]
// Uniform noise with the same variance as normal(0, 1).
const NOISE_SCALE = Math.sqrt(12) / 2
const IMAGE_QUALITY = 0.9
const IMAGE_TIMEOUT_MS = 180_000

interface TvmDevice {
  sync(): Promise<void>
}

interface TvmTensor {
  shape: number[]
  copyFrom(source: TvmTensor | ArrayLike<number>): TvmTensor
  toRawBytes(): Uint8Array
  dispose(): void
}

type TvmFunction = ((...args: unknown[]) => TvmTensor) & { dispose(): void }
type TvmParams = TvmArray & { dispose(): void }

interface TvmVirtualMachine {
  getFunction(name: string): TvmFunction
  getInternalModule(): unknown
  dispose(): void
}

interface WebSdTvm extends TvmInstance {
  cpu(): TvmDevice
  webgpu(): TvmDevice
  initWebGPU(device: GPUDevice): void
  registerInitProgressCallback(callback: (report: { progress: number }) => void): void
  fetchNDArrayCacheInternal(baseUrl: string, records: unknown[], device: TvmDevice, cache: OpfsCache): Promise<void>
  getParamsFromCache(prefix: string, count: number): TvmParams
  createVirtualMachine(device: TvmDevice): TvmVirtualMachine
  asyncLoadWebGPUPiplines(module: unknown): Promise<void>
  attachToCurrentScope<T>(object: T): T
  detachFromCurrentScope<T>(object: T): T
  empty(shape: number[], dtype: string, device: TvmDevice): TvmTensor
  uniform(shape: number[], low: number, high: number, device: TvmDevice): TvmTensor
  dispose(): void
}

interface Tokenizer {
  encode(text: string, addSpecialTokens: boolean): { input_ids: ArrayLike<number> }
}

interface WeightsIndex {
  metadata: { clipParamSize: number; unetParamSize: number; vaeParamSize: number }
  records: { dataPath: string }[]
}

interface SchedulerConsts {
  num_steps: number
  timesteps: number[]
  alpha: number[]
  sigma: number[]
  c0: number[]
  c1: number[]
  c2: number[]
}

declare global {
  interface Window {
    tvmjs: {
      instantiate(wasm: Uint8Array, wasi: unknown): Promise<WebSdTvm>
      detectGPUDevice(): Promise<{ device: GPUDevice } | undefined>
    }
    EmccWASI: new () => unknown
  }
}

const cache = createOpfsCache('stable-diffusion')

const fetchJson = async <T>(url: string): Promise<T> => (await cache.fetchWithCache(url)).json()

const weightUrls = (index: WeightsIndex) => index.records.map((record) => new URL(record.dataPath, WEIGHTS_URL).href)

const isDownloaded = async () => {
  if (!(await cache.has(WEIGHTS_INDEX_URL))) return false
  const urls = [WASM_URL, SCHEDULER_URL, TOKENIZER_URL, ...weightUrls(await fetchJson<WeightsIndex>(WEIGHTS_INDEX_URL))]
  const stored = await Promise.all(urls.map((url) => cache.has(url)))
  return stored.every(Boolean)
}

// tvmjs ships as classic scripts that set window globals.
const loadScript = (src: string) =>
  new Promise<void>((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) return resolve()
    const script = Object.assign(document.createElement('script'), { src })
    script.onload = () => resolve()
    script.onerror = () => reject(new Error(`Failed to load ${src}`))
    document.head.append(script)
  })

const loadTokenizer = async (): Promise<Tokenizer> => {
  // Absolute URL keeps Vite from rewriting the import of a public file.
  const moduleUrl = new URL(`${VENDORED_DIR}${TOKENIZER_MODULE}`, location.href).href
  const { default: init, TokenizerWasm } = await import(/* @vite-ignore */ moduleUrl)
  await init()
  return new TokenizerWasm(await (await cache.fetchWithCache(TOKENIZER_URL)).text())
}

const createRuntime = async () => {
  for (const script of RUNTIME_SCRIPTS) await loadScript(`${VENDORED_DIR}${script}`)
  const wasm = new Uint8Array(await (await cache.fetchWithCache(WASM_URL)).arrayBuffer())
  const tvm = await window.tvmjs.instantiate(wasm, new window.EmccWASI())
  const gpu = await window.tvmjs.detectGPUDevice()
  if (!gpu) throw new Error('This browser does not support WebGPU')
  tvm.initWebGPU(gpu.device)
  return { tvm, device: gpu.device }
}

const createPipeline = (tvm: WebSdTvm, index: WeightsIndex, schedulerConsts: SchedulerConsts) =>
  tvm.withNewScope(() => {
    const keep = <T>(object: T) => tvm.detachFromCurrentScope(object)
    const gpuScalars = (values: number[], dtype: string) =>
      values.map((value) => keep(tvm.empty([], dtype, tvm.webgpu()).copyFrom([value])))
    const vm = keep(tvm.createVirtualMachine(tvm.webgpu()))
    const { metadata } = index
    return {
      vm,
      clip: keep(vm.getFunction('clip')),
      unet: keep(vm.getFunction('unet')),
      vae: keep(vm.getFunction('vae')),
      imageToRgba: keep(vm.getFunction('image_to_rgba')),
      concatEmbeddings: keep(vm.getFunction('concat_embeddings')),
      convertModelOutput: keep(vm.getFunction('dpm_solver_multistep_scheduler_convert_model_output')),
      schedulerStep: keep(vm.getFunction('dpm_solver_multistep_scheduler_step')),
      clipParams: keep(tvm.getParamsFromCache('clip', metadata.clipParamSize)),
      unetParams: keep(tvm.getParamsFromCache('unet', metadata.unetParamSize)),
      vaeParams: keep(tvm.getParamsFromCache('vae', metadata.vaeParamSize)),
      steps: schedulerConsts.num_steps,
      timesteps: gpuScalars(schedulerConsts.timesteps, 'int32'),
      alpha: gpuScalars(schedulerConsts.alpha, 'float32'),
      sigma: gpuScalars(schedulerConsts.sigma, 'float32'),
      c0: gpuScalars(schedulerConsts.c0, 'float32'),
      c1: gpuScalars(schedulerConsts.c1, 'float32'),
      c2: gpuScalars(schedulerConsts.c2, 'float32'),
    }
  })

type Pipeline = ReturnType<typeof createPipeline>

const tokenize = (tvm: WebSdTvm, tokenizer: Tokenizer, prompt: string) => {
  const encoded = Array.from(tokenizer.encode(prompt, true).input_ids).slice(0, MAX_TOKEN_LENGTH)
  const ids = new Int32Array(MAX_TOKEN_LENGTH).fill(encoded[encoded.length - 1]!)
  ids.set(encoded)
  return tvm.empty([1, MAX_TOKEN_LENGTH], 'int32', tvm.webgpu()).copyFrom(ids)
}

// image_to_rgba packs each pixel into one little-endian uint32, so the raw bytes are already RGBA.
const toDataUrl = (rgba: Uint8Array, [height, width]: [number, number]) => {
  const canvas = Object.assign(document.createElement('canvas'), { width, height })
  const pixels = new Uint8ClampedArray(rgba)
  canvas.getContext('2d')!.putImageData(new ImageData(pixels, width, height), 0, 0)
  // ponytail: JPEG data URLs live in localStorage, which caps history near 80 images. Move them to OPFS past that.
  return canvas.toDataURL('image/jpeg', IMAGE_QUALITY)
}

const generateImage = async (tvm: WebSdTvm, tokenizer: Tokenizer, pipeline: Pipeline, prompt: string) => {
  const keep = <T>(object: T) => tvm.detachFromCurrentScope(object)
  const device = tvm.webgpu()
  const start = tvm.withNewScope(() => {
    const positive = pipeline.clip(tokenize(tvm, tokenizer, prompt), pipeline.clipParams)
    const negative = pipeline.clip(tokenize(tvm, tokenizer, ''), pipeline.clipParams)
    return {
      embeddings: keep(pipeline.concatEmbeddings(negative, positive)),
      latents: keep(tvm.uniform(LATENT_SHAPE, -NOISE_SCALE, NOISE_SCALE, device)),
      lastModelOutput: keep(tvm.empty(LATENT_SHAPE, 'float32', device)),
    }
  })
  let { latents, lastModelOutput } = start

  for (let step = 0; step < pipeline.steps; step++) {
    latents = tvm.withNewScope(() => {
      // Attaching the previous step's tensors frees them when this scope ends.
      const previous = tvm.attachToCurrentScope(latents)
      tvm.attachToCurrentScope(lastModelOutput)
      const noise = pipeline.unet(previous, pipeline.timesteps[step], start.embeddings, pipeline.unetParams)
      const modelOutput = pipeline.convertModelOutput(previous, noise, pipeline.alpha[step], pipeline.sigma[step])
      const next = pipeline.schedulerStep(
        previous,
        modelOutput,
        lastModelOutput,
        pipeline.c0[step],
        pipeline.c1[step],
        pipeline.c2[step],
      )
      lastModelOutput = keep(modelOutput)
      return keep(next)
    })
    await device.sync()
  }

  const rgbaOnCpu = tvm.withNewScope(() => {
    tvm.attachToCurrentScope(latents)
    tvm.attachToCurrentScope(lastModelOutput)
    tvm.attachToCurrentScope(start.embeddings)
    const rgba = pipeline.imageToRgba(pipeline.vae(latents, pipeline.vaeParams))
    return keep(tvm.empty(rgba.shape, 'uint32', tvm.cpu()).copyFrom(rgba))
  })
  await device.sync()
  const image = toDataUrl(rgbaOnCpu.toRawBytes(), rgbaOnCpu.shape as [number, number])
  rgbaOnCpu.dispose()
  return image
}

const disposePipeline = (pipeline: Pipeline) => {
  const disposables = [
    pipeline.clip,
    pipeline.unet,
    pipeline.vae,
    pipeline.imageToRgba,
    pipeline.concatEmbeddings,
    pipeline.convertModelOutput,
    pipeline.schedulerStep,
    pipeline.clipParams,
    pipeline.unetParams,
    pipeline.vaeParams,
    ...pipeline.timesteps,
    ...pipeline.alpha,
    ...pipeline.sigma,
    ...pipeline.c0,
    ...pipeline.c1,
    ...pipeline.c2,
    pipeline.vm,
  ]
  disposables.forEach((disposable) => disposable.dispose())
}

export const stableDiffusion: ModelDefinition = {
  name: 'Stable Diffusion',
  isDownloaded,
  load: async (onProgress) => {
    const downloading = !(await isDownloaded())
    const text = downloading ? 'Downloading the weights into OPFS' : 'Loading the weights from OPFS onto the GPU'
    const { tvm, device } = await createRuntime()
    tvm.registerInitProgressCallback((report) => onProgress({ fraction: report.progress, downloading, text }))

    const index = await fetchJson<WeightsIndex>(WEIGHTS_INDEX_URL)
    await tvm.fetchNDArrayCacheInternal(WEIGHTS_URL, index.records, tvm.webgpu(), cache)
    const [schedulerConsts, tokenizer] = await Promise.all([fetchJson<SchedulerConsts>(SCHEDULER_URL), loadTokenizer()])
    const pipeline = createPipeline(tvm, index, schedulerConsts)
    await tvm.asyncLoadWebGPUPiplines(pipeline.vm.getInternalModule())

    return {
      weights: {
        device,
        tensors: readParamTensors(tvm, [pipeline.clipParams, pipeline.unetParams, pipeline.vaeParams]),
        sync: () => tvm.webgpu().sync(),
      },
      respond: async (history: ChatMessage[], reply: ChatMessage) => {
        const prompt = history[history.length - 1]!.content
        reply.image = await withTimeout(
          generateImage(tvm, tokenizer, pipeline, prompt),
          IMAGE_TIMEOUT_MS,
          'Image generation stalled',
        )
      },
      forgetContext: async () => {},
      unload: async () => {
        disposePipeline(pipeline)
        tvm.dispose()
        device.destroy()
      },
    }
  },
}
