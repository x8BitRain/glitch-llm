import type { ElementType, WeightSource, WeightTensor } from '../models/model'
import shaderSource from './corrupt.wgsl?raw'

const WORKGROUP_SIZE = 256
const MAX_WORKGROUPS_PER_DIMENSION = 65535
const WORD_BYTES = 4
const PARAMS_BYTES = 20 * WORD_BYTES
const JITTER_SLOT = 18
// The shader addresses bytes with u32.
const MAX_ADDRESSABLE_BYTES = 2 ** 32
// The undo table holds one byte per changed byte. Past this the GPU can fail without reporting it,
// which strands the weights half corrupted.
const MAX_UNDO_TABLE_BYTES = 2 ** 30
const MODE_CODES = { add: 0, shift: 1, replace: 2, jitter: 3 } as const
const ELEMENT_CODES: Record<ElementType, number> = { other: 0, float16: 1, float32: 2 }
const ELEMENT_BYTES: Record<ElementType, number> = { other: 1, float16: 2, float32: 4 }

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
  jitterPercent: number
}

interface LaidOutTensor extends WeightTensor {
  offset: number
  byteLength: number
}

interface TargetRange {
  start: number
  end: number
  step: number
  firstTarget: number
}

interface TensorTargets {
  firstLocal: number
  step: number
  count: number
}

interface TensorPlan {
  tensor: LaidOutTensor
  targets: TensorTargets
  unitBytes: number
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
  private readonly tensors: LaidOutTensor[]
  private readonly layout: GPUBindGroupLayout
  private readonly computeDeltaPipeline: GPUComputePipeline
  private readonly applyDeltaPipeline: GPUComputePipeline
  private readonly unusedRoots: GPUBuffer
  private readonly weights: WeightSource
  private readonly nextSeed: () => number
  private applied = NO_CORRUPTION

  constructor(weights: WeightSource, nextSeed = randomSeed) {
    this.weights = weights
    this.nextSeed = nextSeed
    this.device = weights.device
    this.tensors = layOutTensors(weights.tensors)
    const last = this.tensors[this.tensors.length - 1]!
    this.totalBytes = last.offset + last.byteLength
    if (this.totalBytes > MAX_ADDRESSABLE_BYTES) {
      throw new Error(`Weights span ${this.totalBytes} bytes, past the corruptor's 32-bit addresses`)
    }

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
    const range = resolveRange(settings, this.totalBytes)
    const plans = changesNothing(settings, range) ? [] : this.plan(settings, range)
    assertUndoTableFits(plans)
    const next = await this.replaceCorruption((encoder) =>
      plans.length > 0 ? this.prepare(settings, range, plans, encoder) : NO_CORRUPTION,
    )
    return next.dispatches.reduce((sum, dispatch) => sum + dispatch.targetCount, 0)
  }

  /** Undoes the current corruption, so the weights match what was loaded. */
  async restore() {
    await this.replaceCorruption(() => NO_CORRUPTION)
  }

  private async replaceCorruption(prepareNext: (encoder: GPUCommandEncoder) => AppliedCorruption) {
    await this.weights.sync()
    this.device.pushErrorScope('out-of-memory')
    this.device.pushErrorScope('validation')
    const encoder = this.device.createCommandEncoder()
    this.runPass(encoder, this.applyDeltaPipeline, this.applied.dispatches)

    const next = prepareNext(encoder)
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
    return next
  }

  private plan(settings: CorruptionSettings, range: TargetRange): TensorPlan[] {
    return this.tensorsIn(range.firstTarget, range.end + 1)
      .filter((tensor) => canCorrupt(settings, tensor))
      .map((tensor) => {
        const unitBytes = chunkBytes(settings, tensor)
        return { tensor, unitBytes, targets: visitChunksOnce(targetsIn(tensor, range), unitBytes) }
      })
      .filter((plan) => plan.targets.count > 0)
  }

  private prepare(
    settings: CorruptionSettings,
    range: TargetRange,
    plans: TensorPlan[],
    encoder: GPUCommandEncoder,
  ): AppliedCorruption {
    const roots = isChainedShift(settings, range) ? this.copyRoots(settings, range, encoder) : undefined
    const corruption: AppliedCorruption = { dispatches: [], buffers: roots ? [roots.buffer] : [] }
    const seed = this.nextSeed()

    for (const { tensor, targets, unitBytes } of plans) {
      const threadCount = targets.count * unitBytes
      const deltas = this.createBuffer(alignUp(threadCount), GPUBufferUsage.STORAGE)
      const params = this.createBuffer(PARAMS_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
      this.device.queue.writeBuffer(params, 0, encodeParams(settings, range, tensor, targets, roots, { seed, unitBytes }))
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
        workgroups: workgroupGrid(threadCount),
        targetCount: threadCount,
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

const layOutTensors = (tensors: WeightTensor[]): LaidOutTensor[] => {
  let offset = 0
  return tensors.map((tensor) => {
    const laidOut = { ...tensor, offset, byteLength: tensor.buffer.size }
    offset += tensor.buffer.size
    return laidOut
  })
}

const resolveRange = (settings: CorruptionSettings, totalBytes: number): TargetRange => {
  const lastByte = totalBytes - 1
  const start = clamp(settings.startByte, 0, lastByte)
  const end = clamp(settings.endByte, start, lastByte)
  const step = Math.max(1, Math.floor(settings.everyNthByte))
  return { start, end, step, firstTarget: start + step }
}

// The original skips these
const changesNothing = (settings: CorruptionSettings, range: TargetRange) => {
  const noEffect = {
    add: settings.addValue === 0,
    shift: settings.shiftBytes === 0,
    replace: settings.replaceFrom === settings.replaceTo,
    jitter: settings.jitterPercent === 0,
  }
  return noEffect[settings.mode] || range.firstTarget > range.end
}

const isChainedShift = (settings: CorruptionSettings, range: TargetRange) => {
  return settings.mode === 'shift' && settings.shiftBytes > 0 && settings.shiftBytes % range.step === 0
}

const targetsIn = (tensor: LaidOutTensor, range: TargetRange): TensorTargets => {
  const lastByte = Math.min(range.end, tensor.offset + tensor.byteLength - 1)
  const firstIndex = Math.max(0, Math.ceil((tensor.offset - range.firstTarget) / range.step))
  const lastIndex = Math.floor((lastByte - range.firstTarget) / range.step)
  return {
    firstLocal: range.firstTarget + firstIndex * range.step - tensor.offset,
    step: range.step,
    count: lastIndex - firstIndex + 1,
  }
}

// A step smaller than the chunk lands in every chunk between the first and last target,
// several times each. Visiting each of those chunks once keeps the undo table small.
const visitChunksOnce = (targets: TensorTargets, unitBytes: number): TensorTargets => {
  if (targets.step >= unitBytes || targets.count <= 0) return targets
  const firstLocal = targets.firstLocal - (targets.firstLocal % unitBytes)
  const lastLocal = targets.firstLocal + (targets.count - 1) * targets.step
  return { firstLocal, step: unitBytes, count: Math.floor((lastLocal - firstLocal) / unitBytes) + 1 }
}

const assertUndoTableFits = (plans: TensorPlan[]) => {
  const undoBytes = plans.reduce((sum, plan) => sum + alignUp(plan.targets.count * plan.unitBytes), 0)
  if (undoBytes <= MAX_UNDO_TABLE_BYTES) return
  const needed = (undoBytes / 2 ** 30).toFixed(1)
  throw new Error(`This selection needs a ${needed} GB undo table, over the 1 GB limit. Raise "Corrupt every" or shrink the range.`)
}

const encodeParams = (
  settings: CorruptionSettings,
  range: TargetRange,
  tensor: LaidOutTensor,
  targets: TensorTargets,
  roots: Roots | undefined,
  { seed, unitBytes }: { seed: number; unitBytes: number },
) => {
  const params = new Uint32Array(PARAMS_BYTES / WORD_BYTES)
  params.set([
    targets.firstLocal,
    targets.step,
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
    unitBytes,
    targets.count * unitBytes,
    seed,
    ELEMENT_CODES[tensor.elementType],
  ])
  new Float32Array(params.buffer)[JITTER_SLOT] = settings.jitterPercent / 100
  return params
}

// Jitter does arithmetic on whole floats, so integer and packed tensors sit it out.
const canCorrupt = (settings: CorruptionSettings, tensor: LaidOutTensor) =>
  settings.mode !== 'jitter' || tensor.elementType !== 'other'

// Jitter rewrites whole floats. Every other mode works on single bytes.
const chunkBytes = (settings: CorruptionSettings, tensor: LaidOutTensor) =>
  settings.mode === 'jitter' ? ELEMENT_BYTES[tensor.elementType] : 1

const randomSeed = () => crypto.getRandomValues(new Uint32Array(1))[0]!

const workgroupGrid = (targetCount: number): [number, number] => {
  const groups = Math.ceil(targetCount / WORKGROUP_SIZE)
  const columns = Math.min(groups, MAX_WORKGROUPS_PER_DIMENSION)
  return [columns, Math.ceil(groups / columns)]
}

const clamp = (value: number, min: number, max: number) => {
  return Math.min(Math.max(value, min), max)
}

const wrapByte = (value: number) => {
  return ((value % 256) + 256) % 256
}

const alignDown = (bytes: number) => {
  return Math.floor(bytes / WORD_BYTES) * WORD_BYTES
}

const alignUp = (bytes: number) => {
  return Math.max(WORD_BYTES, Math.ceil(bytes / WORD_BYTES) * WORD_BYTES)
}
