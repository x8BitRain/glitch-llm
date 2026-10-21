import { ref } from 'vue'

const TOAST_DURATION_MS = 5000

export const toastMessage = ref('')

let hideTimer: ReturnType<typeof setTimeout> | undefined

export const showToast = (message: string) => {
  toastMessage.value = message
  clearTimeout(hideTimer)
  hideTimer = setTimeout(() => (toastMessage.value = ''), TOAST_DURATION_MS)
}
