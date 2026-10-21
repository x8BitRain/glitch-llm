import { CreateMLCEngine, hasModelInCache, type AppConfig, type MLCEngine } from '@mlc-ai/web-llm'
import type { ChatMessage } from '../stores/chats'
import { RecoverableError, readParamTensors, withTimeout, type ModelDefinition, type TvmArray, type TvmInstance } from './model'

const MODEL_ID = 'TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC'
const APP_CONFIG: AppConfig = {
  cacheBackend: 'opfs',
  model_list: [
    {
      model: 'https://huggingface.co/mlc-ai/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC',
      model_id: MODEL_ID,
      model_lib:
        'https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_86/base/TinyLlama-1.1B-Chat-v1.0-q4f16_1_cs1k-webgpu.wasm',
      required_features: ['shader-f16'],
      overrides: { context_window_size: 2048 },
    },
  ],
}

const MAX_REPLY_TOKENS = 512
const STALL_TIMEOUT_MS = 30_000
// WebLLM reports one progress stream. Text switches to this prefix once every shard sits in OPFS.
const GPU_LOAD_PREFIX = 'Loading model from cache'

interface EngineInternals {
  loadedModelIdToPipeline: Map<string, { tvm: TvmInstance; params: TvmArray; sync(): Promise<void> }>
}

type ChunkIterator = AsyncIterator<{ choices: { delta: { content?: string | null } }[] }>

const nextChunk = (chunks: ChunkIterator) =>
  withTimeout(chunks.next(), STALL_TIMEOUT_MS, 'No output for a while, generation stalled')

// Garbage text is the point, so only errors, stalls and silence count as a crash.
const streamReply = async (engine: MLCEngine, history: ChatMessage[], reply: ChatMessage) => {
  try {
    const stream = await engine.chat.completions.create({ messages: history, stream: true, max_tokens: MAX_REPLY_TOKENS })
    const chunks = stream[Symbol.asyncIterator]()
    for (let next = await nextChunk(chunks); !next.done; next = await nextChunk(chunks)) {
      reply.content += next.value.choices[0]?.delta.content ?? ''
    }
  } catch (error) {
    if (error instanceof Error && error.name === 'ContextWindowSizeExceededError') {
      throw new RecoverableError('This chat no longer fits in the 2k context. Start a new chat.')
    }
    throw error
  }
  if (!reply.content) throw new Error('The model produced no output')
}

export const tinyLlama: ModelDefinition = {
  name: 'TinyLlama',
  isDownloaded: () => hasModelInCache(MODEL_ID, APP_CONFIG),
  load: async (onProgress) => {
    const engine = await CreateMLCEngine(MODEL_ID, {
      appConfig: APP_CONFIG,
      initProgressCallback: (report) =>
        onProgress({ fraction: report.progress, downloading: !report.text.startsWith(GPU_LOAD_PREFIX), text: report.text }),
    })
    const pipeline = (engine as unknown as EngineInternals).loadedModelIdToPipeline.get(MODEL_ID)!
    return {
      weights: {
        device: pipeline.tvm.lib.webGPUContext.device,
        tensors: readParamTensors(pipeline.tvm, [pipeline.params]),
        sync: () => pipeline.sync(),
      },
      respond: (history, reply) => streamReply(engine, history, reply),
      // Drops the KV cache built with the old weights, so the next reply runs fully on the new ones.
      forgetContext: () => engine.resetChat(),
      unload: () => engine.unload(),
    }
  },
}
