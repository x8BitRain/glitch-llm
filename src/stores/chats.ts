import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

const TITLE_LENGTH = 40

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface Chat {
  id: string
  title: string
  messages: ChatMessage[]
}

export const useChatStore = defineStore(
  'chats',
  () => {
    const chats = ref<Chat[]>([])
    const activeChatId = ref<string | null>(null)
    const activeChat = computed(() => chats.value.find((chat) => chat.id === activeChatId.value))

    function startNewChat() {
      activeChatId.value = null
    }

    function openChat(id: string) {
      activeChatId.value = id
    }

    function deleteChat(id: string) {
      chats.value = chats.value.filter((chat) => chat.id !== id)
      if (activeChatId.value === id) activeChatId.value = null
    }

    function addMessage(message: ChatMessage): ChatMessage {
      if (!activeChat.value) {
        const chat = { id: crypto.randomUUID(), title: message.content.slice(0, TITLE_LENGTH), messages: [] }
        chats.value.unshift(chat)
        activeChatId.value = chat.id
      }
      const messages = activeChat.value!.messages
      messages.push(message)
      return messages[messages.length - 1]!
    }

    return { chats, activeChatId, activeChat, startNewChat, openChat, deleteChat, addMessage }
  },
  { persist: true },
)
