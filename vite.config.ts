import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'

// GitHub Pages serves the site from x8bitrain.github.io/glitch-llm/.
const PAGES_BASE = '/glitch-llm/'

// Only `bun dev` serves from the root. Build and preview both match Pages.
export default defineConfig(({ command, isPreview }) => ({
  base: command === 'serve' && !isPreview ? '/' : PAGES_BASE,
  plugins: [
    vue({
      template: {
        compilerOptions: {
          isCustomElement: (tag) => tag.startsWith('md-'),
        },
      },
    }),
  ],
}))
