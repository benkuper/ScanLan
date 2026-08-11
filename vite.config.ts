import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

const host = process.env.TAURI_DEV_HOST;

export default defineConfig({
  plugins: [svelte()],
  clearScreen: false,
  // Spark is a ~5 MB ESM bundle and is dynamically imported only when a splat
  // is opened. Keep it out of the cold-start prebundle.
  optimizeDeps: {
    // Vite otherwise discovers every HTML file below the repository root.
    // Native dependency trees include thousands of generated documentation
    // pages, which can exhaust Windows file handles during `npm run debug`.
    entries: ['index.html'],
    exclude: ['@sparkjsdev/spark']
  },
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: 'ws',
          host,
          port: 1421
        }
      : undefined,
    watch: {
      // These trees contain generated runtimes, native build products, and
      // captured media. Watching them can mean crawling tens of gigabytes
      // before the renderer receives its first module.
      ignored: [
        '**/build/**',
        '**/dist/**',
        '**/Log/**',
        '**/native/**',
        '**/splat-worker/**',
        '**/src-tauri/**',
        '**/test-photos/**',
        '**/worker/**'
      ]
    }
  },
  envPrefix: ['VITE_', 'TAURI_'],
  build: {
    target: process.env.TAURI_ENV_PLATFORM === 'windows' ? 'chrome105' : 'safari13',
    sourcemap: Boolean(process.env.TAURI_ENV_DEBUG),
    chunkSizeWarningLimit: 700
  }
});
