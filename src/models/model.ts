import type { ChatMessage } from '../stores/chats'

export type ElementType = 'float16' | 'float32' | 'other'

export interface WeightTensor {
  buffer: GPUBuffer
  elementType: ElementType
}

/** The weight tensors the corruptor treats as one ROM, in order. */
export interface WeightSource {
  device: GPUDevice
  tensors: WeightTensor[]
  sync(): Promise<void>
}

export interface LoadProgress {
  fraction: number
  downloading: boolean
  text: string
}

export interface LoadedModel {
  readonly weights: WeightSource
  /** Fills `reply` from the chat so far. Throws when the model crashes. */
  respond(history: ChatMessage[], reply: ChatMessage): Promise<void>
  /** Drops any state built with the old weights after a corruption. */
  forgetContext(): Promise<void>
  unload(): Promise<void>
}

export interface ModelDefinition {
  readonly name: string
  isDownloaded(): Promise<boolean>
  load(onProgress: (progress: LoadProgress) => void): Promise<LoadedModel>
}

/** A failure the user can fix themselves, so the model keeps running. */
export class RecoverableError extends Error {}

// Untyped tvmjs internals. WebLLM's tvmjs and the older Web SD tvmjs share these names.
export interface TvmInstance {
  lib: { webGPUContext: { device: GPUDevice; gpuBufferFromPtr(pointer: number): GPUBuffer } }
  withNewScope<T>(action: () => T): T
}

interface TvmTensorHandle {
  dataPtr: number
  dlDataType: { code: number; bits: number; lanes: number }
}

export interface TvmArray {
  size(): number
  get(index: number): TvmTensorHandle
}

const DLPACK_FLOAT = 2

// Packed and quantized tensors (TinyLlama's 4-bit weights) store integers, so they count as other.
const elementTypeOf = ({ dlDataType }: TvmTensorHandle): ElementType => {
  if (dlDataType.code !== DLPACK_FLOAT || dlDataType.lanes !== 1) return 'other'
  if (dlDataType.bits === 16) return 'float16'
  return dlDataType.bits === 32 ? 'float32' : 'other'
}

export const readParamTensors = (tvm: TvmInstance, paramArrays: TvmArray[]): WeightTensor[] =>
  tvm.withNewScope(() =>
    paramArrays.flatMap((params) =>
      Array.from({ length: params.size() }, (_, index) => {
        const tensor = params.get(index)
        return { buffer: tvm.lib.webGPUContext.gpuBufferFromPtr(tensor.dataPtr), elementType: elementTypeOf(tensor) }
      }),
    ),
  )

export const withTimeout = <T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> => {
  let timer: ReturnType<typeof setTimeout> | undefined
  const expiry = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(message)), timeoutMs)
  })
  return Promise.race([promise, expiry]).finally(() => clearTimeout(timer))
}

export const errorMessage = (error: unknown) => (error instanceof Error ? error.message : String(error))
