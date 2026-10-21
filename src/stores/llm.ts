import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { CreateMLCEngine, hasModelInCache, type AppConfig, type MLCEngine } from '@mlc-ai/web-llm'
import { GpuCorruptor, type ChatPipelineInternals, type CorruptionSettings } from '../gpu/corruptor'
import { showToast } from '../toast'
import type { ChatMessage } from './chats'

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
const CORRUPT_TIMEOUT_MS = 60_000
const UNLOAD_TIMEOUT_MS = 5_000

export type LlmStatus = 'checking' | 'absent' | 'loading' | 'ready' | 'generating' | 'corrupting' | 'failed'

interface EngineInternals {
  loadedModelIdToPipeline: Map<string, ChatPipelineInternals>
}

export const useLlmStore = defineStore('llm', () => {
  const status = ref<LlmStatus>('checking')
  const progress = ref(0)
  const progressText = ref('')
  const modelBytes = ref(0)
  const isReady = computed(() => status.value === 'ready')
  const isDownloading = computed(() => !progressText.value.startsWith('Loading model from cache'))

  let engine: MLCEngine | undefined
  let corruptor: GpuCorruptor | undefined

  async function loadIfDownloaded() {
    const downloaded = await hasModelInCache(MODEL_ID, APP_CONFIG).catch(() => false)
    if (downloaded) await load()
    else status.value = 'absent'
  }

  async function load() {
    status.value = 'loading'
    progress.value = 0
    progressText.value = ''
    try {
      engine = await CreateMLCEngine(MODEL_ID, {
        appConfig: APP_CONFIG,
        initProgressCallback: (report) => {
          progress.value = report.progress
          progressText.value = report.text
        },
      })
      const pipeline = (engine as unknown as EngineInternals).loadedModelIdToPipeline.get(MODEL_ID)!
      corruptor = new GpuCorruptor(pipeline)
      modelBytes.value = corruptor.totalBytes
      status.value = 'ready'
      showToast('TinyLlama is ready to chat')
    } catch (error) {
      status.value = 'failed'
      showToast(`Model failed to load: ${errorMessage(error)}`)
    }
  }

  // restart on errors, stalls, or zero output.
  async function reply(messages: ChatMessage[], onText: (text: string) => void) {
    status.value = 'generating'
    let producedText = false
    try {
      const chunks = await engine!.chat.completions.create({ messages, stream: true, max_tokens: MAX_REPLY_TOKENS })
      for await (const chunk of withStallTimeout(chunks, STALL_TIMEOUT_MS)) {
        const text = chunk.choices[0]?.delta.content ?? ''
        producedText ||= text.length > 0
        onText(text)
      }
    } catch (error) {
      if (isContextOverflow(error)) {
        status.value = 'ready'
        showToast('This chat no longer fits in the 2k context. Start a new chat.')
        return
      }
      return restart(`WebLLM crashed: ${errorMessage(error)}`)
    }
    if (!producedText) return restart('The model produced no output')
    status.value = 'ready'
  }

  async function corrupt(settings: CorruptionSettings) {
    status.value = 'corrupting'
    try {
      const targetedBytes = await withTimeout(corruptor!.corrupt(settings), CORRUPT_TIMEOUT_MS, 'Corruption timed out')
      // Drops the KV cache built with the old weights, so the next reply runs fully on the new ones.
      await engine!.resetChat()
      showToast(`Corruption finished. ${targetedBytes.toLocaleString()} bytes targeted.`)
    } catch (error) {
      showToast(errorMessage(error))
    }
    status.value = 'ready'
  }

  async function restart(reason: string) {
    showToast(`${reason}. Restarting model.`)
    const crashedEngine = engine
    engine = undefined
    corruptor = undefined
    await withTimeout(crashedEngine?.unload() ?? Promise.resolve(), UNLOAD_TIMEOUT_MS, 'Unload timed out').catch(
      () => undefined,
    )
    await load()
  }

  return { status, progress, progressText, modelBytes, isReady, isDownloading, loadIfDownloaded, load, reply, corrupt }
})

async function* withStallTimeout<T>(stream: AsyncIterable<T>, timeoutMs: number) {
  const iterator = stream[Symbol.asyncIterator]()
  while (true) {
    const result = await withTimeout(iterator.next(), timeoutMs, 'No output for a while, generation stalled')
    if (result.done) return
    yield result.value
  }
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined
  const expiry = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(message)), timeoutMs)
  })
  return Promise.race([promise, expiry]).finally(() => clearTimeout(timer))
}

function isContextOverflow(error: unknown) {
  return error instanceof Error && error.name === 'ContextWindowSizeExceededError'
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error)
}
