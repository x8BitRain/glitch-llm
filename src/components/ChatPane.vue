<script setup lang="ts">
import { computed, nextTick, ref, useTemplateRef, watch } from 'vue'
import { useChatStore, type ChatMode } from '../stores/chats'
import { useModelStore, type ModelStatus } from '../stores/model'

const PLACEHOLDERS: Record<Exclude<ModelStatus, 'ready'>, string> = {
  checking: 'Looking for a saved model...',
  absent: 'Load the model to start chatting',
  loading: 'Loading the model...',
  failed: 'Load the model to start chatting',
  generating: 'Model is responding...',
  corrupting: 'Corrupting the weights...',
  restoring: 'Restoring the original weights...',
}
const READY_PLACEHOLDERS: Record<ChatMode, string> = {
  text: 'Send a message',
  image: 'Describe an image',
}
const EMPTY_CHAT_TEXT: Record<ChatMode, string> = {
  text: 'Ask me Anything',
  image: 'Imagine Anything',
}

const chats = useChatStore()
const model = useModelStore()
const draft = ref('')
const messageList = useTemplateRef('messageList')
const placeholder = computed(() =>
  model.status === 'ready' ? READY_PLACEHOLDERS[chats.mode] : PLACEHOLDERS[model.status],
)

watch(
  () => [chats.activeChat?.messages.at(-1)?.content, chats.activeChat?.messages.at(-1)?.image],
  async () => {
    await nextTick()
    messageList.value?.scrollTo({ top: messageList.value.scrollHeight })
  },
)

const isPending = (index: number) =>
  model.status === 'generating' && index === chats.activeChat!.messages.length - 1

const send = async () => {
  const prompt = draft.value.trim()
  if (!prompt || !model.isReady) return
  draft.value = ''
  chats.addMessage({ role: 'user', content: prompt })
  const history = chats.activeChat!.messages.map(({ role, content }) => ({ role, content }))
  const reply = chats.addMessage({ role: 'assistant', content: '' })
  await model.respond(history, reply)
}
</script>

<template>
  <section class="chat">
    <div ref="messageList" class="messages">
      <p v-if="!chats.activeChat" class="empty md-typescale-headline-small">{{ EMPTY_CHAT_TEXT[chats.mode] }}</p>
      <template v-for="(message, index) in chats.activeChat?.messages" :key="index">
        <div v-if="message.role === 'user'" class="bubble md-typescale-body-large">{{ message.content }}</div>
        <div v-else class="answer md-typescale-body-large">
          <img v-if="message.image" class="image" :src="message.image" :alt="chats.activeChat!.messages[index - 1]?.content" />
          <template v-else-if="message.content">{{ message.content }}</template>
          <md-circular-progress v-else-if="isPending(index)" indeterminate class="typing" />
          <span v-else class="no-output">(no output)</span>
        </div>
      </template>
    </div>
    <form class="composer" @submit.prevent="send">
      <md-outlined-text-field
        class="prompt"
        type="textarea"
        rows="2"
        :label="placeholder"
        :value="draft"
        :disabled="!model.isReady"
        @input="draft = ($event.target as HTMLTextAreaElement).value"
        @keydown.enter.exact.prevent="send"
      />
      <md-icon-button type="submit" aria-label="Send" :disabled="!model.isReady || !draft.trim()">
        <md-icon>send</md-icon>
      </md-icon-button>
    </form>
  </section>
</template>

<style scoped>
.chat {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-height: 0;
}

.messages {
  display: flex;
  flex: 1;
  flex-direction: column;
  gap: 16px;
  overflow-y: auto;
  padding: 24px max(24px, calc((100% - 760px) / 2));
}

.empty {
  margin: auto;
  color: var(--md-sys-color-on-surface-variant);
}

.bubble {
  align-self: flex-end;
  max-width: 75%;
  padding: 10px 16px;
  border-radius: 20px;
  white-space: pre-wrap;
  color: var(--md-sys-color-on-primary-container);
  background: var(--md-sys-color-primary-container);
}

.answer {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.image {
  display: block;
  width: min(512px, 100%);
  border-radius: 12px;
}

.typing {
  --md-circular-progress-size: 24px;
}

.no-output {
  color: var(--md-sys-color-on-surface-variant);
}

.composer {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px max(24px, calc((100% - 760px) / 2)) 24px;
}

.prompt {
  flex: 1;
  resize: none;
}
</style>
