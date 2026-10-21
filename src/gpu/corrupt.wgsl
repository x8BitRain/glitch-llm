// Vinesauce ROM Corruptor (https://github.com/Rikerz/VRC) byte corruption on one weight tensor.

struct Params {
  firstTarget: u32,
  step: u32,
  count: u32,
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
  padding: u32,
}

const WORKGROUP_SIZE = 256u;
const MODE_ADD = 0u;
const MODE_SHIFT = 1u;
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

fn shiftDestination(position: u32) -> i32 {
  let destinationGlobal = i32(params.tensorOffset + position) + params.shift;
  let inRange = destinationGlobal >= i32(params.rangeStart) && destinationGlobal <= i32(params.rangeEnd);
  let destination = destinationGlobal - i32(params.tensorOffset);
  let inTensor = destination >= 0 && destination < i32(params.tensorBytes);
  return select(NO_BYTE, destination, inRange && inTensor);
}

fn changedByte(k: u32) -> i32 {
  let position = targetPosition(k);
  if (params.mode == MODE_SHIFT) {
    return shiftDestination(position);
  }
  return i32(position);
}

fn shiftSourceByte(position: u32) -> u32 {
  if (params.chained == 0u) {
    return readWeightByte(position);
  }
  let fromOrigin = (params.tensorOffset + position - params.chainOrigin) % u32(params.shift);
  let rootIndex = params.chainOrigin + fromOrigin - params.rootsBase;
  return byteOf(roots[rootIndex >> 2u], rootIndex);
}

fn corruptedByte(position: u32, original: u32) -> u32 {
  switch params.mode {
    case MODE_ADD: {
      return (original + params.addValue) & 0xffu;
    }
    case MODE_SHIFT: {
      return shiftSourceByte(position);
    }
    default: {
      return select(original, params.replaceTo, original == params.replaceFrom);
    }
  }
}

fn targetIndex(id: vec3<u32>, groups: vec3<u32>) -> u32 {
  return id.x + id.y * groups.x * WORKGROUP_SIZE;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn computeDelta(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let k = targetIndex(id, groups);
  if (k >= params.count) {
    return;
  }
  let changed = changedByte(k);
  if (changed == NO_BYTE) {
    return;
  }
  let original = readWeightByte(u32(changed));
  let delta = original ^ corruptedByte(targetPosition(k), original);
  atomicOr(&deltas[k >> 2u], delta << ((k & 3u) * 8u));
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn applyDelta(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let k = targetIndex(id, groups);
  if (k >= params.count) {
    return;
  }
  let changed = changedByte(k);
  if (changed == NO_BYTE) {
    return;
  }
  let position = u32(changed);
  let delta = byteOf(atomicLoad(&deltas[k >> 2u]), k);
  atomicXor(&weights[position >> 2u], delta << ((position & 3u) * 8u));
}
