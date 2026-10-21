// Vinesauce ROM Corruptor (https://github.com/Rikerz/VRC) byte corruption on one weight tensor.

struct Params {
  firstTarget: u32,
  step: u32,
  mode: u32,
  addValue: u32,
  shift: i32,
  replaceFrom: u32,
  replaceTo: u32,
  tensorOffset: u32,
  tensorBytes: u32,
  rangeStart: u32,
  rangeEnd: u32,
  chainOrigin: u32,
  rootsBase: u32,
  chained: u32,
  unitBytes: u32,
  threadCount: u32,
  seed: u32,
  elementKind: u32,
  jitter: f32,
  padding: u32,
}

const WORKGROUP_SIZE = 256u;
const MODE_ADD = 0u;
const MODE_SHIFT = 1u;
const MODE_REPLACE = 2u;
const MODE_JITTER = 3u;
const ELEMENT_FLOAT32 = 2u;
const FLOAT16_MAX = 65504.0;
const UNIT_INTERVAL_SCALE = 2.3283064e-10;
const NO_BYTE = -1;

@group(0) @binding(0) var<storage, read_write> weights: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> deltas: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> roots: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn byteOf(word: u32, position: u32) -> u32 {
  return (word >> ((position & 3u) * 8u)) & 0xffu;
}

fn readWeightByte(position: u32) -> u32 {
  return byteOf(atomicLoad(&weights[position >> 2u]), position);
}

fn targetPosition(k: u32) -> u32 {
  return params.firstTarget + k * params.step;
}

// Global addresses stay u32 because Stable Diffusion's weights run past 2^31 bytes.
fn shiftDestination(position: u32) -> i32 {
  let sourceGlobal = params.tensorOffset + position;
  let distance = u32(abs(params.shift));
  let leavesAddressSpace = select(sourceGlobal > 0xffffffffu - distance, distance > sourceGlobal, params.shift < 0);
  if (leavesAddressSpace) {
    return NO_BYTE;
  }
  let destinationGlobal = select(sourceGlobal + distance, sourceGlobal - distance, params.shift < 0);
  let inRange = destinationGlobal >= params.rangeStart && destinationGlobal <= params.rangeEnd;
  let inTensor = destinationGlobal >= params.tensorOffset && destinationGlobal - params.tensorOffset < params.tensorBytes;
  return select(NO_BYTE, i32(destinationGlobal - params.tensorOffset), inRange && inTensor);
}

// Byte `b` of the aligned chunk that target k sits in.
fn chunkByte(k: u32, b: u32) -> i32 {
  let position = targetPosition(k);
  let chunkStart = position - position % params.unitBytes;
  let chunkPosition = chunkStart + b;
  return select(NO_BYTE, i32(chunkPosition), chunkPosition < params.tensorBytes);
}

fn changedByte(k: u32, b: u32) -> i32 {
  switch params.mode {
    case MODE_SHIFT: {
      return shiftDestination(targetPosition(k));
    }
    case MODE_JITTER: {
      return chunkByte(k, b);
    }
    default: {
      return i32(targetPosition(k));
    }
  }
}

fn shiftSourceByte(position: u32) -> u32 {
  if (params.chained == 0u) {
    return readWeightByte(position);
  }
  let fromOrigin = (params.tensorOffset + position - params.chainOrigin) % u32(params.shift);
  let rootIndex = params.chainOrigin + fromOrigin - params.rootsBase;
  return byteOf(roots[rootIndex >> 2u], rootIndex);
}

// PCG hash, the source of jitter's random factors.
fn hash(value: u32) -> u32 {
  let state = value * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

// Uniform in [-1, 1), the same for every byte of one float.
fn noise(elementStart: u32) -> f32 {
  return f32(hash(params.seed ^ hash(params.tensorOffset + elementStart))) * UNIT_INTERVAL_SCALE * 2.0 - 1.0;
}

// Scales the whole float by 1 ± jitter, then hands back this thread's byte of the result.
// Working on values keeps every weight finite, unlike byte edits that hit the exponent.
fn jitteredByte(position: u32) -> u32 {
  let word = atomicLoad(&weights[position >> 2u]);
  let elementStart = position - position % params.unitBytes;
  let factor = 1.0 + params.jitter * noise(elementStart);
  if (params.elementKind == ELEMENT_FLOAT32) {
    return byteOf(bitcast<u32>(bitcast<f32>(word) * factor), position);
  }
  var halves = unpack2x16float(word);
  let lane = (elementStart >> 1u) & 1u;
  halves[lane] = clamp(halves[lane] * factor, -FLOAT16_MAX, FLOAT16_MAX);
  return byteOf(pack2x16float(halves), position);
}

fn corruptedByte(k: u32, b: u32, original: u32) -> u32 {
  switch params.mode {
    case MODE_ADD: {
      return (original + params.addValue) & 0xffu;
    }
    case MODE_SHIFT: {
      return shiftSourceByte(targetPosition(k));
    }
    case MODE_JITTER: {
      return jitteredByte(u32(chunkByte(k, b)));
    }
    default: {
      return select(original, params.replaceTo, original == params.replaceFrom);
    }
  }
}

fn threadIndex(id: vec3<u32>, groups: vec3<u32>) -> u32 {
  return id.x + id.y * groups.x * WORKGROUP_SIZE;
}

// One thread per byte of each target's chunk. Jitter's chunk is one float, the other modes use single bytes.
@compute @workgroup_size(WORKGROUP_SIZE)
fn computeDelta(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let slot = threadIndex(id, groups);
  if (slot >= params.threadCount) {
    return;
  }
  let k = slot / params.unitBytes;
  let b = slot % params.unitBytes;
  let changed = changedByte(k, b);
  if (changed == NO_BYTE) {
    return;
  }
  let original = readWeightByte(u32(changed));
  let delta = original ^ corruptedByte(k, b, original);
  atomicOr(&deltas[slot >> 2u], delta << ((slot & 3u) * 8u));
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn applyDelta(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let slot = threadIndex(id, groups);
  if (slot >= params.threadCount) {
    return;
  }
  let changed = changedByte(slot / params.unitBytes, slot % params.unitBytes);
  if (changed == NO_BYTE) {
    return;
  }
  let position = u32(changed);
  let delta = byteOf(atomicLoad(&deltas[slot >> 2u]), slot);
  atomicXor(&weights[position >> 2u], delta << ((position & 3u) * 8u));
}
