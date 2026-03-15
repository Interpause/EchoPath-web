import { defineConfig } from "vite";
import react, { reactCompilerPreset } from "@vitejs/plugin-react";
import { viteStaticCopy } from "vite-plugin-static-copy";
import babel from "@rolldown/plugin-babel";
import path from "node:path";

// https://vite.dev/config/
export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  plugins: [
    react(),
    babel({ presets: [reactCompilerPreset()] }),
    viteStaticCopy({
      targets: [
        {
          src: "node_modules/@litertjs/core/wasm/*",
          dest: "litert",
        },
      ],
    }),
  ],
});
