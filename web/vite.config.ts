import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// 开发时界面跑在 5173，RPC 与状态推送经代理转发到本机的 Speech Studio（FastAPI）。
const target = process.env.VITE_API_TARGET ?? 'http://127.0.0.1:8123';

export default defineConfig({
  plugins: [react()],
  base: './',
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': { target, changeOrigin: true },
      '/ws': { target, ws: true, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        // 稳定的 React 运行时独立缓存，业务改动无需重新加载这一部分。
        manualChunks: { react: ['react', 'react-dom', 'react-dom/client'] },
      },
    },
  },
});
