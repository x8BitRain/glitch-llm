<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { CorruptionMode } from '../gpu/corruptor'
import { useChatStore } from '../stores/chats'
import { useModelStore, type ModelStatus } from '../stores/model'
import { showToast } from '../toast'

const HELP = {
  range:
    'The first and last byte to consider for corruption, in hexadecimal. Bytes count across every weight tensor laid end to end. The config, tokenizer and tensor metadata sit outside this range, so the model always loads again.',
  increment: "The value that '+' and '-' buttons will change the start and end bytes by, in hexadecimal.",
  autoEnd: 'When checked, corruption will go until the end of the model.',
  everyNthByte:
    'This decimal value specifies how often to corrupt bytes in the model. A value of one will corrupt every byte, a value of two will corrupt every second byte, and so on.',
  add: 'When this option is selected, this decimal value is added to each byte selected for corruption.',
  shift:
    'When this option is selected, each byte selected for corruption is shifted right by the number of bytes specified by this decimal value. This value can be negative, resulting in a shift to the left.',
  replace:
    'When this option is selected, each byte selected for corruption is compared to the first hexadecimal value. If it matches, it is replaced by the second hexadecimal value.',
  jitter:
    'When this option is selected, each weight selected for corruption is multiplied by a random factor within this many percent of 1, so 50 scales it anywhere from 0.5x to 1.5x. It works on the numbers rather than their bytes, so weights never blow up. Only floating point weights take part. Every CORRUPT! press rolls new factors. Good starting points: TinyLlama at every 1st byte, ±35%. Stable Diffusion at every 16th byte, ±50%.',
}
// Hand-tuned for Stable Diffusion's fp32 weights. The corruptor also starts out on these.
const SWEETSPOT = { everyNthByte: 90, mode: 'jitter', jitterPercent: 100 } as const
const BYTE_MAX = 0xff
const WEIGHT_CHANGE_LABELS: Partial<Record<ModelStatus, string>> = {
  corrupting: 'Corrupting the weights...',
  restoring: 'Restoring the original weights...',
}

interface FieldFormat {
  parse(text: string): number | undefined
  print(value: number): string
}

const HEX: FieldFormat = {
  parse: (text) => (/^[0-9a-f]+$/i.test(text) ? Number.parseInt(text, 16) : undefined),
  print: (value) => value.toString(16).toUpperCase(),
}
const HEX_BYTE: FieldFormat = {
  parse: (text) => {
    const value = HEX.parse(text)
    return value !== undefined && value <= BYTE_MAX ? value : undefined
  },
  print: HEX.print,
}
const INTEGER: FieldFormat = {
  parse: (text) => (/^-?\d+$/.test(text) ? Number.parseInt(text, 10) : undefined),
  print: String,
}
const POSITIVE_INTEGER: FieldFormat = {
  parse: (text) => {
    const value = INTEGER.parse(text)
    return value !== undefined && value > 0 ? value : undefined
  },
  print: String,
}

const model = useModelStore()
const chats = useChatStore()

const startByte = ref(0)
const endByte = ref(0x1000)
const increment = ref(0x1000)
const autoEnd = ref(true)
const everyNthByte = ref<number>(SWEETSPOT.everyNthByte)
const mode = ref<CorruptionMode>(SWEETSPOT.mode)
const addValue = ref(25)
const shiftBytes = ref(0)
const replaceFrom = ref(0)
const replaceTo = ref(0)
const jitterPercent = ref<number>(SWEETSPOT.jitterPercent)
const help = ref<string>()

const lastByte = computed(() => model.modelBytes - 1)
const controlsDisabled = computed(() => model.modelBytes === 0)
const endLocked = computed(() => controlsDisabled.value || autoEnd.value)
const loadingLabel = computed(
  () => `${model.progress.downloading ? 'Downloading' : 'Loading onto GPU'} ${Math.floor(model.progress.fraction * 100)}%`,
)

// Every edit funnels through here, so the range never leaves the weight bytes.
const keepRangeInModel = () => {
  startByte.value = clamp(startByte.value, 0, lastByte.value)
  endByte.value = autoEnd.value ? lastByte.value : clamp(endByte.value, startByte.value, lastByte.value)
}

// Size drops to zero while models swap. Clamping to that would push the range negative.
watch([() => model.modelBytes, autoEnd], () => {
  if (model.modelBytes > 0) keepRangeInModel()
})

const raiseStart = () => {
  startByte.value = Math.min(startByte.value + increment.value, endByte.value)
  keepRangeInModel()
}

const lowerStart = () => {
  startByte.value = Math.max(startByte.value - increment.value, 0)
  keepRangeInModel()
}

const raiseEnd = () => {
  endByte.value += increment.value
  keepRangeInModel()
}

const lowerEnd = () => {
  endByte.value = Math.max(endByte.value - increment.value, startByte.value)
  keepRangeInModel()
}

const raiseRange = () => {
  const width = endByte.value - startByte.value
  endByte.value = Math.min(endByte.value + increment.value, lastByte.value)
  startByte.value = Math.min(startByte.value + increment.value, endByte.value - width)
  keepRangeInModel()
}

const lowerRange = () => {
  const width = endByte.value - startByte.value
  startByte.value = Math.max(startByte.value - increment.value, 0)
  endByte.value = startByte.value === 0 ? width : endByte.value - increment.value
  keepRangeInModel()
}

// Shows the invalid text the way the original did, then puts the last good value back.
const readField = (event: Event, current: number, format: FieldFormat, name: string) => {
  const field = event.target as HTMLInputElement
  const value = format.parse(field.value.trim())
  if (value !== undefined) return value
  showToast(`Invalid ${name}.`)
  field.value = format.print(current)
  return current
}

const updateStart = (event: Event) => {
  startByte.value = readField(event, startByte.value, HEX, 'start byte')
  keepRangeInModel()
}

const updateEnd = (event: Event) => {
  endByte.value = readField(event, endByte.value, HEX, 'end byte')
  keepRangeInModel()
}

const clamp = (value: number, min: number, max: number) => {
  return Math.min(Math.max(value, min), max)
}

const corrupt = () => {
  model.corrupt({
    startByte: startByte.value,
    endByte: endByte.value,
    everyNthByte: everyNthByte.value,
    mode: mode.value,
    addValue: addValue.value,
    shiftBytes: shiftBytes.value,
    replaceFrom: replaceFrom.value,
    replaceTo: replaceTo.value,
    jitterPercent: jitterPercent.value,
  })
}

const applySweetspot = () => {
  everyNthByte.value = SWEETSPOT.everyNthByte
  mode.value = SWEETSPOT.mode
  jitterPercent.value = SWEETSPOT.jitterPercent
  corrupt()
}
</script>

<template>
  <aside class="corruptor">
    <h2 class="md-typescale-title-large">Corruptor</h2>
    <p class="caption md-typescale-body-small">
      <template v-if="model.modelBytes">Weights: 0 to {{ HEX.print(lastByte) }} ({{ model.modelBytes.toLocaleString() }} bytes)</template>
      <template v-else>Load the model to unlock the controls.</template>
    </p>

    <div class="range">
      <div class="range-fields">
        <div class="row">
          <md-outlined-text-field
            label="Start (hex)"
            :value="HEX.print(startByte)"
            :disabled="controlsDisabled"
            @change="updateStart"
          />
          <md-icon-button aria-label="Raise start byte" :disabled="controlsDisabled" @click="raiseStart">
            <md-icon>add</md-icon>
          </md-icon-button>
          <md-icon-button aria-label="Lower start byte" :disabled="controlsDisabled" @click="lowerStart">
            <md-icon>remove</md-icon>
          </md-icon-button>
        </div>
        <div class="row">
          <md-outlined-text-field
            label="End (hex)"
            :value="HEX.print(endByte)"
            :disabled="endLocked"
            @change="updateEnd"
          />
          <md-icon-button aria-label="Raise end byte" :disabled="endLocked" @click="raiseEnd">
            <md-icon>add</md-icon>
          </md-icon-button>
          <md-icon-button aria-label="Lower end byte" :disabled="endLocked" @click="lowerEnd">
            <md-icon>remove</md-icon>
          </md-icon-button>
        </div>
      </div>
      <div class="range-shift">
        <md-filled-tonal-button aria-label="Move range up" :disabled="endLocked" @click="raiseRange">
          <md-icon>add</md-icon>
        </md-filled-tonal-button>
        <md-filled-tonal-button aria-label="Move range down" :disabled="endLocked" @click="lowerRange">
          <md-icon>remove</md-icon>
        </md-filled-tonal-button>
      </div>
      <md-icon-button aria-label="Range help" @click="help = HELP.range"><md-icon>help</md-icon></md-icon-button>
    </div>

    <div class="row">
      <md-outlined-text-field
        class="short"
        label="Increment (hex)"
        :value="HEX.print(increment)"
        :disabled="controlsDisabled"
        @change="increment = readField($event, increment, HEX, 'increment')"
      />
      <md-icon-button aria-label="Increment help" @click="help = HELP.increment"><md-icon>help</md-icon></md-icon-button>
      <label class="check">
        <md-checkbox :checked="autoEnd" :disabled="controlsDisabled" @change="autoEnd = ($event.target as HTMLInputElement).checked" />
        Auto End
      </label>
      <md-icon-button aria-label="Auto end help" @click="help = HELP.autoEnd"><md-icon>help</md-icon></md-icon-button>
    </div>

    <div class="row">
      <md-outlined-text-field
        class="short"
        label="Corrupt every"
        suffix-text="th byte"
        :value="POSITIVE_INTEGER.print(everyNthByte)"
        :disabled="controlsDisabled"
        @change="everyNthByte = readField($event, everyNthByte, POSITIVE_INTEGER, 'byte interval')"
      />
      <md-icon-button aria-label="Every nth byte help" @click="help = HELP.everyNthByte">
        <md-icon>help</md-icon>
      </md-icon-button>
    </div>

    <div class="row">
      <md-radio name="mode" aria-label="Add" :checked="mode === 'add'" :disabled="controlsDisabled" @change="mode = 'add'" />
      <md-outlined-text-field
        class="short"
        label="Add"
        suffix-text="to byte"
        :value="INTEGER.print(addValue)"
        :disabled="controlsDisabled"
        @change="addValue = readField($event, addValue, INTEGER, 'add value')"
      />
      <md-icon-button aria-label="Add help" @click="help = HELP.add"><md-icon>help</md-icon></md-icon-button>
    </div>

    <div class="row">
      <md-radio
        name="mode"
        aria-label="Shift right"
        :checked="mode === 'shift'"
        :disabled="controlsDisabled"
        @change="mode = 'shift'"
      />
      <md-outlined-text-field
        class="short"
        label="Shift right"
        suffix-text="bytes"
        :value="INTEGER.print(shiftBytes)"
        :disabled="controlsDisabled"
        @change="shiftBytes = readField($event, shiftBytes, INTEGER, 'shift')"
      />
      <md-icon-button aria-label="Shift help" @click="help = HELP.shift"><md-icon>help</md-icon></md-icon-button>
    </div>

    <div class="row">
      <md-radio
        name="mode"
        aria-label="Replace"
        :checked="mode === 'replace'"
        :disabled="controlsDisabled"
        @change="mode = 'replace'"
      />
      <md-outlined-text-field
        class="tiny"
        label="Replace (hex)"
        :value="HEX.print(replaceFrom)"
        :disabled="controlsDisabled"
        @change="replaceFrom = readField($event, replaceFrom, HEX_BYTE, 'byte to replace')"
      />
      <md-outlined-text-field
        class="tiny"
        label="with (hex)"
        :value="HEX.print(replaceTo)"
        :disabled="controlsDisabled"
        @change="replaceTo = readField($event, replaceTo, HEX_BYTE, 'replacement byte')"
      />
      <md-icon-button aria-label="Replace help" @click="help = HELP.replace"><md-icon>help</md-icon></md-icon-button>
    </div>

    <div class="row">
      <md-radio name="mode" aria-label="Jitter" :checked="mode === 'jitter'" :disabled="controlsDisabled" @change="mode = 'jitter'" />
      <md-outlined-text-field
        class="short"
        label="Jitter weights by"
        prefix-text="±"
        suffix-text="%"
        :value="POSITIVE_INTEGER.print(jitterPercent)"
        :disabled="controlsDisabled"
        @change="jitterPercent = readField($event, jitterPercent, POSITIVE_INTEGER, 'jitter amount')"
      />
      <md-icon-button aria-label="Jitter help" @click="help = HELP.jitter"><md-icon>help</md-icon></md-icon-button>
    </div>

    <div class="action">
      <template v-if="model.status === 'checking'">
        <md-linear-progress indeterminate />
        <span class="md-typescale-body-medium">Looking for a saved model...</span>
      </template>
      <template v-else-if="model.status === 'loading'">
        <md-linear-progress :value="model.progress.fraction" />
        <span class="md-typescale-label-large">{{ loadingLabel }}</span>
        <span class="caption md-typescale-body-small">{{ model.progress.text }}</span>
      </template>
      <md-filled-button v-else-if="model.status === 'absent' || model.status === 'failed'" @click="model.load()">
        <md-icon slot="icon">download</md-icon>
        Load {{ model.modelName }}
      </md-filled-button>
      <template v-else>
        <md-filled-button class="corrupt-button" :disabled="!model.isReady" @click="corrupt">
          CORRUPT!
        </md-filled-button>
        <md-filled-tonal-button v-if="chats.mode === 'image'" :disabled="!model.isReady" @click="applySweetspot">
          <md-icon slot="icon">tune</md-icon>
          Apply sweetspot
        </md-filled-tonal-button>
        <md-outlined-button :disabled="!model.isReady || !model.isCorrupted" @click="model.restore()">
          <md-icon slot="icon">restore</md-icon>
          Restore original model
        </md-outlined-button>
        <md-linear-progress v-if="model.status in WEIGHT_CHANGE_LABELS" indeterminate />
        <span class="ready md-typescale-label-large">
          <md-icon>check_circle</md-icon>
          {{ WEIGHT_CHANGE_LABELS[model.status] ?? (model.isCorrupted ? 'model is corrupted' : 'model is loaded') }}
        </span>
      </template>
    </div>

    <md-dialog :open="help !== undefined" @closed="help = undefined">
      <div slot="headline">Help</div>
      <div slot="content">{{ help }}</div>
      <div slot="actions">
        <md-text-button @click="help = undefined">OK</md-text-button>
      </div>
    </md-dialog>
  </aside>
</template>

<style scoped>
.corruptor {
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  gap: 12px;
  width: 400px;
  overflow-y: auto;
  padding: 16px;
  background: var(--md-sys-color-surface-container);
  --md-outlined-text-field-top-space: 10px;
  --md-outlined-text-field-bottom-space: 10px;
}

h2 {
  margin: 0;
}

.caption {
  margin: 0;
  color: var(--md-sys-color-on-surface-variant);
}

.row {
  display: flex;
  align-items: center;
  gap: 4px;
}

.row md-outlined-text-field {
  flex: 1;
  min-width: 0;
}

.row .short {
  flex: 0 1 180px;
}

.row .tiny {
  flex: 0 1 120px;
}

.range {
  display: flex;
  align-items: center;
  gap: 4px;
}

.range-fields {
  display: flex;
  flex: 1;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
}

.range-shift {
  display: flex;
  gap: 4px;
  align-self: stretch;
}

.range-shift md-filled-tonal-button {
  height: auto;
  min-width: 40px;
  --md-filled-tonal-button-leading-space: 8px;
  --md-filled-tonal-button-trailing-space: 8px;
}

.check {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-left: auto;
}

.action {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 8px;
}

.corrupt-button {
  height: 56px;
  --md-filled-button-container-color: var(--md-sys-color-error);
  --md-filled-button-label-text-color: var(--md-sys-color-on-error);
  --md-filled-button-icon-color: var(--md-sys-color-on-error);
  --md-filled-button-label-text-size: 20px;
}

.ready {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--md-sys-color-on-surface-variant);
}
</style>
