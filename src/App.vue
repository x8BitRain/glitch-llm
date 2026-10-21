<script setup lang="ts">
import { onMounted, ref } from 'vue'
import ChatPane from './components/ChatPane.vue'
import CorruptorPanel from './components/CorruptorPanel.vue'
import HistoryDrawer from './components/HistoryDrawer.vue'
import { useChatStore } from './stores/chats'
import { useLlmStore } from './stores/llm'
import { toastMessage } from './toast'

const chats = useChatStore()
const llm = useLlmStore()
const isHistoryOpen = ref(false)

onMounted(llm.loadIfDownloaded)
</script>

<template>
  <div class="layout">
    <HistoryDrawer v-if="isHistoryOpen" />
    <main class="main">
      <header class="top-bar">
        <md-icon-button aria-label="Toggle chat history" @click="isHistoryOpen = !isHistoryOpen">
          <md-icon>menu</md-icon>
        </md-icon-button>
        <md-filled-tonal-button @click="chats.startNewChat()">
          <md-icon slot="icon">add</md-icon>
          New chat
        </md-filled-tonal-button>
        <span class="title md-typescale-title-medium">{{ chats.activeChat?.title ?? 'Glitch LLM' }}</span>
      </header>
      <ChatPane />
    </main>
    <CorruptorPanel />
  </div>
  <div v-if="toastMessage" class="snackbar md-typescale-body-medium" role="status">{{ toastMessage }}</div>
</template>

<style scoped>
.layout {
  display: flex;
  height: 100vh;
}

.main {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-width: 0;
}

.top-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 16px;
}

.title {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.snackbar {
  position: fixed;
  bottom: 24px;
  left: 50%;
  max-width: 560px;
  padding: 14px 16px;
  border-radius: 4px;
  color: var(--md-sys-color-inverse-on-surface);
  background: var(--md-sys-color-inverse-surface);
  box-shadow: 0 3px 6px rgb(0 0 0 / 25%);
  transform: translateX(-50%);
}
</style>
