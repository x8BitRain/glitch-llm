<script setup lang="ts">
import { nextTick, ref, useTemplateRef, watch } from 'vue'
import { useChatStore } from '../stores/chats'
import { useLlmStore, type LlmStatus } from '../stores/llm'

const PLACEHOLDERS: Record<LlmStatus, string> = {
  checking: 'Looking for a saved model...',
  absent: 'Load the LLM to start chatting',
  loading: 'Loading the LLM...',
  failed: 'Load the LLM to start chatting',
  ready: 'Send a message',
  generating: 'Model is responding...',
  corrupting: 'Corrupting the weights...',
}

const chats = useChatStore()
const llm = useLlmStore()
const draft = ref('')
const messageList = useTemplateRef('messageList')

watch(
  () => chats.activeChat?.messages.at(-1)?.content,
  async () => {
    await nextTick()
    messageList.value?.scrollTo({ top: messageList.value.scrollHeight })
  },
)

async function send() {
  const prompt = draft.value.trim()
  if (!prompt || !llm.isReady) return
  draft.value = ''
  chats.addMessage({ role: 'user', content: prompt })
  const history = chats.activeChat!.messages.map(({ role, content }) => ({ role, content }))
  const reply = chats.addMessage({ role: 'assistant', content: '' })
  await llm.reply(history, (text) => (reply.content += text))
}
</script>

<template>
  <section class="chat">
    <div ref="messageList" class="messages">
      <p v-if="!chats.activeChat" class="empty md-typescale-headline-small">Ask me Anything</p>
      <template v-for="(message, index) in chats.activeChat?.messages" :key="index">
        <div v-if="message.role === 'user'" class="bubble md-typescale-body-large">{{ message.content }}</div>
        <div v-else class="answer md-typescale-body-large">
          <template v-if="message.content">{{ message.content }}</template>
          <md-circular-progress v-else-if="llm.status === 'generating'" indeterminate class="typing" />
          <span v-else class="no-output">(no output)</span>
        </div>
      </template>
    </div>
    <form class="composer" @submit.prevent="send">
      <md-outlined-text-field
        class="prompt"
        type="textarea"
        rows="2"
        :label="PLACEHOLDERS[llm.status]"
        :value="draft"
        :disabled="!llm.isReady"
        @input="draft = ($event.target as HTMLTextAreaElement).value"
        @keydown.enter.exact.prevent="send"
      />
      <md-icon-button type="submit" aria-label="Send" :disabled="!llm.isReady || !draft.trim()">
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
