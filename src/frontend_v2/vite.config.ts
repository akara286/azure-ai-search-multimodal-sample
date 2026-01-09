import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// https://vite.dev/config/
export default defineConfig({
    plugins: [react()],
    optimizeDeps: {
        esbuildOptions: {
            target: "esnext"
        }
    },
    build: {
        outDir: "../backend_v2/static",
        emptyOutDir: true,
        sourcemap: true,
        target: "esnext"
    },
    server: {
        port: 5174,
        proxy: {
            "/v2": {
                target: "http://localhost:8765"
            }
        }
    }
});
