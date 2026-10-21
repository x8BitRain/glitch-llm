import { defineStore } from 'pinia'
import { computed, ref, shallowRef } from 'vue'
import { GpuCorruptor, type CorruptionSettings } from '../gpu/corruptor'
import { errorMessage, RecoverableError, withTimeout, type LoadedModel, type LoadProgress } from '../models/model'
import { stableDiffusion } from '../models/stableDiffusion'
import { tinyLlama } from '../models/tinyLlama'
import { showToast } from '../toast'
import type { ChatMessage, ChatMode } from './chats'

const MODELS = { text: tinyLlama, image: stableDiffusion }
const CORRUPT_TIMEOUT_MS = 60_000
const UNLOAD_TIMEOUT_MS = 5_000
const NO_PROGRESS: LoadProgress = { fraction: 0, downloading: true, text: '' }

export type ModelStatus = 'checking' | 'absent' | 'loading' | 'ready' | 'generating' | 'corrupting' | 'restoring' | 'failed'

// One model on the GPU at a time. Both together would not fit on most machines.
export const useModelStore = defineStore('model', () => {
  const status = ref<ModelStatus>('checking')
  const progress = ref(NO_PROGRESS)
  const modelBytes = ref(0)
  const isCorrupted = ref(false)
  const definition = shallowRef(MODELS.text)
  const modelName = computed(() => definition.value.name)
  const isReady = computed(() => status.value === 'ready')
  const isIdle = computed(() => ['ready', 'absent', 'failed'].includes(status.value))

  let model: LoadedModel | undefined
  let corruptor: GpuCorruptor | undefined

  const activate = async (mode: ChatMode) => {
    // Busy before unloading, so nothing can start a load on the outgoing model.
    status.value = 'checking'
    await unload()
    definition.value = MODELS[mode]
    await loadIfDownloaded()
  }

  const loadIfDownloaded = async () => {
    const downloaded = await definition.value.isDownloaded().catch(() => false)
    if (downloaded) await load()
    else status.value = 'absent'
  }

  const load = async () => {
    status.value = 'loading'
    progress.value = NO_PROGRESS
    try {
      model = await definition.value.load((report) => (progress.value = report))
      corruptor = new GpuCorruptor(model.weights)
      modelBytes.value = corruptor.totalBytes
      status.value = 'ready'
      showToast('Model is ready to chat')
    } catch (error) {
      status.value = 'failed'
      showToast(`Model failed to load: ${errorMessage(error)}`)
    }
  }

  const respond = async (history: ChatMessage[], reply: ChatMessage) => {
    status.value = 'generating'
    try {
      await model!.respond(history, reply)
      status.value = 'ready'
    } catch (error) {
      if (error instanceof RecoverableError) {
        status.value = 'ready'
        showToast(error.message)
        return
      }
      await restart(`${modelName.value} crashed: ${errorMessage(error)}`)
    }
  }

  const changeWeights = async (busyStatus: ModelStatus, change: (corruptor: GpuCorruptor) => Promise<string>) => {
    status.value = busyStatus
    try {
      const message = await withTimeout(change(corruptor!), CORRUPT_TIMEOUT_MS, 'Changing the weights timed out')
      await model!.forgetContext()
      showToast(message)
    } catch (error) {
      showToast(errorMessage(error))
    }
    status.value = 'ready'
  }

  const corrupt = (settings: CorruptionSettings) =>
    changeWeights('corrupting', async (weights) => {
      const targetedBytes = await weights.corrupt(settings)
      isCorrupted.value = targetedBytes > 0
      return `Corruption finished. ${targetedBytes.toLocaleString()} bytes targeted.`
    })

  const restore = () =>
    changeWeights('restoring', async (weights) => {
      await weights.restore()
      isCorrupted.value = false
      return 'Model restored to its original weights.'
    })

  const restart = async (reason: string) => {
    showToast(`${reason}. Restarting model.`)
    await unload()
    await load()
  }

  const unload = async () => {
    const loaded = model
    model = undefined
    corruptor = undefined
    modelBytes.value = 0
    isCorrupted.value = false
    await withTimeout(loaded?.unload() ?? Promise.resolve(), UNLOAD_TIMEOUT_MS, 'Unload timed out').catch(
      () => undefined,
    )
  }

  return { status, progress, modelBytes, modelName, isCorrupted, isReady, isIdle, activate, load, respond, corrupt, restore }
})
