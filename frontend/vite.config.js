import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Port dédié (5173/5174 sont souvent déjà pris par d'autres projets Vite)
export default defineConfig({
  plugins: [react()],
  server: { port: Number(process.env.PORT) || 5190, strictPort: true },
  preview: { port: Number(process.env.PORT) || 5190, strictPort: true },
});
