import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '')
  return {
    plugins: [react()],
    server: {
      // FastAPI serves every route under /api, so no rewrite and no CORS in dev.
      proxy: { '/api': { target: env.VITE_BACKEND_URL || 'http://127.0.0.1:8080', changeOrigin: true } },
    },
  }
})
