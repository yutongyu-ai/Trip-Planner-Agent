import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Fail loudly if 5173 is taken instead of silently drifting to 5174+ --
    // the backend's CORS allowlist (backend/app/main.py) only accepts 5173,
    // so a silent port change looks like a working dev server until the
    // first API call fails with a CORS error.
    strictPort: true,
  },
})
