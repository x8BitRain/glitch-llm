import shaderSource from './corrupt.wgsl?raw'

const WORKGROUP_SIZE = 256
const MAX_WORKGROUPS_PER_DIMENSION = 65535
const WORD_BYTES = 4
const PARAMS_BYTES = 16 * WORD_BYTES
const MODE_CODES = { add: 0, shift: 1, replace: 2 } as const

export type CorruptionMode = keyof typeof MODE_CODES

export interface CorruptionSettings {
  startByte: number
  endByte: number
  everyNthByte: number
  mode: CorruptionMode
  addValue: number
  shiftBytes: number
  replaceFrom: number
  replaceTo: number
}

// these are untyped WebLLM and tvmjs internals we need to get the webGPU context directly.
export interface ChatPipelineInternals {
  tvm: {
    lib: { webGPUContext: { device: GPUDevice; gpuBufferFromPtr(pointer: number): GPUBuffer } }
    withNewScope<T>(action: () => T): T
  }
  params: { size(): number; get(index: number): { dataPtr: number } }
  sync(): Promise<void>
}

interface WeightTensor {
  buffer: GPUBuffer
  offset: number
  byteLength: number
}

interface TargetRange {
  start: number
  end: number
  step: number
  firstTarget: number
}

interface Roots {
  buffer: GPUBuffer
  base: number
}

interface TensorDispatch {
  bindGroup: GPUBindGroup
  workgroups: [number, number]
  targetCount: number
}

interface AppliedCorruption {
  dispatches: TensorDispatch[]
  buffers: GPUBuffer[]
}

const NO_CORRUPTION: AppliedCorruption = { dispatches: [], buffers: [] }


//  Corrupts the loaded weights while in GPU memory.

export class GpuCorruptor {
  readonly totalBytes: number
  private readonly device: GPUDevice
  private readonly tensors: WeightTensor[]
  private readonly layout: GPUBindGroupLayout
  private readonly computeDeltaPipeline: GPUComputePipeline
  private readonly applyDeltaPipeline: GPUComputePipeline
  private readonly unusedRoots: GPUBuffer
  private readonly pipeline: ChatPipelineInternals
  private applied = NO_CORRUPTION

  constructor(pipeline: ChatPipelineInternals) {
    this.pipeline = pipeline
    this.device = pipeline.tvm.lib.webGPUContext.device
    this.tensors = readWeightTensors(pipeline)
    const last = this.tensors[this.tensors.length - 1]!
    this.totalBytes = last.offset + last.byteLength

    this.layout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    })
    const module = this.device.createShaderModule({ code: shaderSource })
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.layout] })
    this.computeDeltaPipeline = this.device.createComputePipeline({
      layout,
      compute: { module, entryPoint: 'computeDelta' },
    })
    this.applyDeltaPipeline = this.device.createComputePipeline({
      layout,
      compute: { module, entryPoint: 'applyDelta' },
    })
    this.unusedRoots = this.createBuffer(WORD_BYTES, GPUBufferUsage.STORAGE)
  }

  // Undoes the previous corruption, then applies this one. Returns how many bytes it targeted.
  async corrupt(settings: CorruptionSettings): Promise<number> {
    await this.pipeline.sync()
    this.device.pushErrorScope('out-of-memory')
    this.device.pushErrorScope('validation')
    const encoder = this.device.createCommandEncoder()
    this.runPass(encoder, this.applyDeltaPipeline, this.applied.dispatches)

    const range = resolveRange(settings, this.totalBytes)
    const next = changesNothing(settings, range) ? NO_CORRUPTION : this.prepare(settings, range, encoder)
    this.runPass(encoder, this.computeDeltaPipeline, next.dispatches)
    this.runPass(encoder, this.applyDeltaPipeline, next.dispatches)

    this.device.queue.submit([encoder.finish()])
    // A failed submit runs nothing, not even the undo, so the old corruption stays valid.
    const validationError = await this.device.popErrorScope()
    const memoryError = await this.device.popErrorScope()
    const error = validationError ?? memoryError
    if (error) {
      next.buffers.forEach((buffer) => buffer.destroy())
      throw new Error(`GPU rejected the corruption: ${error.message}`)
    }
    await this.device.queue.onSubmittedWorkDone()
    this.applied.buffers.forEach((buffer) => buffer.destroy())
    this.applied = next
    return next.dispatches.reduce((sum, dispatch) => sum + dispatch.targetCount, 0)
  }

  private prepare(settings: CorruptionSettings, range: TargetRange, encoder: GPUCommandEncoder): AppliedCorruption {
    const roots = isChainedShift(settings, range) ? this.copyRoots(settings, range, encoder) : undefined
    const corruption: AppliedCorruption = { dispatches: [], buffers: roots ? [roots.buffer] : [] }

    for (const tensor of this.tensorsIn(range.firstTarget, range.end + 1)) {
      const targets = targetsIn(tensor, range)
      if (targets.count <= 0) continue
      const deltas = this.createBuffer(alignUp(targets.count), GPUBufferUsage.STORAGE)
      const params = this.createBuffer(PARAMS_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
      this.device.queue.writeBuffer(params, 0, encodeParams(settings, range, tensor, targets, roots))
      corruption.buffers.push(deltas, params)
      corruption.dispatches.push({
        bindGroup: this.device.createBindGroup({
          layout: this.layout,
          entries: [
            { binding: 0, resource: { buffer: tensor.buffer } },
            { binding: 1, resource: { buffer: deltas } },
            { binding: 2, resource: { buffer: roots?.buffer ?? this.unusedRoots } },
            { binding: 3, resource: { buffer: params } },
          ],
        }),
        workgroups: workgroupGrid(targets.count),
        targetCount: targets.count,
      })
    }
    return corruption
  }

  private copyRoots(settings: CorruptionSettings, range: TargetRange, encoder: GPUCommandEncoder): Roots {
    const base = alignDown(range.firstTarget)
    const end = alignUp(Math.min(range.firstTarget + settings.shiftBytes, this.totalBytes))
    const buffer = this.createBuffer(end - base, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)
    for (const tensor of this.tensorsIn(base, end)) {
      const from = Math.max(base, tensor.offset)
      const to = Math.min(end, tensor.offset + tensor.byteLength)
      encoder.copyBufferToBuffer(tensor.buffer, from - tensor.offset, buffer, from - base, to - from)
    }
    return { buffer, base }
  }

  private runPass(encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, dispatches: TensorDispatch[]) {
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    for (const dispatch of dispatches) {
      pass.setBindGroup(0, dispatch.bindGroup)
      pass.dispatchWorkgroups(...dispatch.workgroups)
    }
    pass.end()
  }

  private tensorsIn(start: number, endExclusive: number) {
    return this.tensors.filter((tensor) => tensor.offset < endExclusive && tensor.offset + tensor.byteLength > start)
  }

  private createBuffer(size: number, usage: GPUBufferUsageFlags) {
    return this.device.createBuffer({ size, usage })
  }
}

function readWeightTensors(pipeline: ChatPipelineInternals): WeightTensor[] {
  const { tvm, params } = pipeline
  return tvm.withNewScope(() => {
    const tensors: WeightTensor[] = []
    let offset = 0
    for (let index = 0; index < params.size(); index++) {
      const buffer = tvm.lib.webGPUContext.gpuBufferFromPtr(params.get(index).dataPtr)
      tensors.push({ buffer, offset, byteLength: buffer.size })
      offset += buffer.size
    }
    return tensors
  })
}

function resolveRange(settings: CorruptionSettings, totalBytes: number): TargetRange {
  const lastByte = totalBytes - 1
  const start = clamp(settings.startByte, 0, lastByte)
  const end = clamp(settings.endByte, start, lastByte)
  const step = Math.max(1, Math.floor(settings.everyNthByte))
  return { start, end, step, firstTarget: start + step }
}

// The original skips these
function changesNothing(settings: CorruptionSettings, range: TargetRange) {
  const noEffect = {
    add: settings.addValue === 0,
    shift: settings.shiftBytes === 0,
    replace: settings.replaceFrom === settings.replaceTo,
  }
  return noEffect[settings.mode] || range.firstTarget > range.end
}

function isChainedShift(settings: CorruptionSettings, range: TargetRange) {
  return settings.mode === 'shift' && settings.shiftBytes > 0 && settings.shiftBytes % range.step === 0
}

function targetsIn(tensor: WeightTensor, range: TargetRange) {
  const lastByte = Math.min(range.end, tensor.offset + tensor.byteLength - 1)
  const firstIndex = Math.max(0, Math.ceil((tensor.offset - range.firstTarget) / range.step))
  const lastIndex = Math.floor((lastByte - range.firstTarget) / range.step)
  return {
    firstLocal: range.firstTarget + firstIndex * range.step - tensor.offset,
    count: lastIndex - firstIndex + 1,
  }
}

function encodeParams(
  settings: CorruptionSettings,
  range: TargetRange,
  tensor: WeightTensor,
  targets: ReturnType<typeof targetsIn>,
  roots: Roots | undefined,
) {
  return new Uint32Array([
    targets.firstLocal,
    range.step,
    targets.count,
    MODE_CODES[settings.mode],
    wrapByte(settings.addValue),
    settings.shiftBytes >>> 0,
    settings.replaceFrom,
    settings.replaceTo,
    tensor.offset,
    tensor.byteLength,
    range.start,
    range.end,
    range.firstTarget,
    roots?.base ?? 0,
    roots ? 1 : 0,
    0,
  ])
}

function workgroupGrid(targetCount: number): [number, number] {
  const groups = Math.ceil(targetCount / WORKGROUP_SIZE)
  const columns = Math.min(groups, MAX_WORKGROUPS_PER_DIMENSION)
  return [columns, Math.ceil(groups / columns)]
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max)
}

function wrapByte(value: number) {
  return ((value % 256) + 256) % 256
}

function alignDown(bytes: number) {
  return Math.floor(bytes / WORD_BYTES) * WORD_BYTES
}

function alignUp(bytes: number) {
  return Math.max(WORD_BYTES, Math.ceil(bytes / WORD_BYTES) * WORD_BYTES)
}
