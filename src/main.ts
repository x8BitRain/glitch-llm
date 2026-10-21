import { createApp } from 'vue'
import { createPinia } from 'pinia'
import piniaPluginPersistedstate from 'pinia-plugin-persistedstate'
import { styles as typescaleStyles } from '@material/web/typography/md-typescale-styles.js'
import '@material/web/button/filled-button.js'
import '@material/web/button/filled-tonal-button.js'
import '@material/web/button/text-button.js'
import '@material/web/checkbox/checkbox.js'
import '@material/web/dialog/dialog.js'
import '@material/web/icon/icon.js'
import '@material/web/iconbutton/icon-button.js'
import '@material/web/list/list-item.js'
import '@material/web/progress/circular-progress.js'
import '@material/web/progress/linear-progress.js'
import '@material/web/radio/radio.js'
import '@material/web/textfield/outlined-text-field.js'
import './style.css'
import App from './App.vue'

document.adoptedStyleSheets.push(typescaleStyles.styleSheet!)

const pinia = createPinia()
pinia.use(piniaPluginPersistedstate)

createApp(App).use(pinia).mount('#app')
