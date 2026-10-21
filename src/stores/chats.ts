import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

const TITLE_LENGTH = 40

export type ChatMode = 'text' | 'image'

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  image?: string
}

export interface Chat {
  id: string
  title: string
  mode: ChatMode
  messages: ChatMessage[]
}

export const useChatStore = defineStore(
  'chats',
  () => {
    const mode = ref<ChatMode>('text')
    const chats = ref<Chat[]>([])
    const activeChatIds = ref<Record<ChatMode, string | null>>({ text: null, image: null })
    const modeChats = computed(() => chats.value.filter((chat) => chat.mode === mode.value))
    const activeChat = computed(() => modeChats.value.find((chat) => chat.id === activeChatIds.value[mode.value]))

    const switchMode = (next: ChatMode) => {
      mode.value = next
    }

    const startNewChat = () => {
      activeChatIds.value[mode.value] = null
    }

    const openChat = (id: string) => {
      activeChatIds.value[mode.value] = id
    }

    const deleteChat = (id: string) => {
      chats.value = chats.value.filter((chat) => chat.id !== id)
      if (activeChatIds.value[mode.value] === id) startNewChat()
    }

    // First message of a new chat creates the chat, so empty chats never pile up in history.
    const addMessage = (message: ChatMessage): ChatMessage => {
      if (!activeChat.value) {
        const title = message.content.slice(0, TITLE_LENGTH)
        const chat: Chat = { id: crypto.randomUUID(), title, mode: mode.value, messages: [] }
        chats.value.unshift(chat)
        openChat(chat.id)
      }
      const messages = activeChat.value!.messages
      messages.push(message)
      return messages[messages.length - 1]!
    }

    return { mode, chats, activeChatIds, modeChats, activeChat, switchMode, startNewChat, openChat, deleteChat, addMessage }
  },
  {
    persist: {
      // Chats saved before image mode existed have no mode and were all text.
      afterHydrate: ({ store }) => store.chats.forEach((chat: Chat) => (chat.mode ??= 'text')),
    },
  },
)
