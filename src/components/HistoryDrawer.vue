<script setup lang="ts">
import { useChatStore } from '../stores/chats'

const chats = useChatStore()
</script>

<template>
  <nav class="drawer">
    <h2 class="heading md-typescale-title-small">Chats</h2>
    <p v-if="chats.modeChats.length === 0" class="empty md-typescale-body-medium">No chats yet.</p>
    <div
      v-for="chat in chats.modeChats"
      :key="chat.id"
      class="chat-row"
      :class="{ active: chat.id === chats.activeChat?.id }"
    >
      <md-list-item type="button" class="chat-link" @click="chats.openChat(chat.id)">
        {{ chat.title }}
      </md-list-item>
      <md-icon-button aria-label="Delete chat" @click="chats.deleteChat(chat.id)">
        <md-icon>delete</md-icon>
      </md-icon-button>
    </div>
  </nav>
</template>

<style scoped>
.drawer {
  width: 280px;
  flex-shrink: 0;
  overflow-y: auto;
  padding: 12px;
  background: var(--md-sys-color-surface-container-low);
  --md-list-item-container-color: transparent;
}

.heading,
.empty {
  margin: 8px 16px;
  color: var(--md-sys-color-on-surface-variant);
}

.chat-row {
  display: flex;
  align-items: center;
  overflow: hidden;
  border-radius: 28px;
}

.chat-link {
  flex: 1;
  min-width: 0;
}

.active {
  background: var(--md-sys-color-secondary-container);
}
</style>
