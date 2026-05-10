import process from 'node:process'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const usePolling = process.env.VITE_USE_POLLING === 'true'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  ...(usePolling
    ? {
        server: {
          watch: {
            usePolling: true,
            interval: 100,
          },
        },
      }
    : {}),
})
