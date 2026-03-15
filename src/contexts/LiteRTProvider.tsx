import type { PropsWithChildren } from "react";
import { useEffect, useState, useRef } from "react";
import { LiteRTContext } from "./LiteRTContext";
import {
  CompileOptions,
  loadAndCompile,
  loadLiteRt,
  getWebGpuDevice,
} from "@litertjs/core";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu";
import { WebGPUBackend } from "@tensorflow/tfjs-backend-webgpu";

async function loadRuntime() {
  // Initialize TensorFlow.js WebGPU backend
  await tf.setBackend("webgpu");

  // Initialize LiteRT.js's Wasm files
  await loadLiteRt(`/litert/`);

  // Make TFJS use the same GPU device as LiteRT.js (for tensor conversion)
  const device = await getWebGpuDevice();
  tf.removeBackend("webgpu");
  if (!device) {
    console.error("No WebGPU device available, falling back to CPU...");
    tf.setBackend("wasm");
    return;
  }
  tf.registerBackend("webgpu", () => new WebGPUBackend(device));
  await tf.setBackend("webgpu");
}

async function loadYOLOModel() {}

export function LiteRTProvider({ children }: PropsWithChildren) {
  const [model, setModel] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<unknown | null>(null);

  // Use a ref to prevent double-initialization in React Strict Mode
  const isInitializing = useRef(false);

  useEffect(() => {
    if (isInitializing.current) return;
    isInitializing.current = true;

    async function init() {
      try {
        // Replace with your actual model loading logic
        await loadRuntime();
        await loadYOLOModel();
        setModel(null);
      } catch (err) {
        setError(err);
      } finally {
        setIsLoading(false);
      }
    }

    init();
  }, []);

  // 3. Memoize the value so consumers don't re-render unnecessarily
  const value = { model, isLoading, error };

  return (
    <LiteRTContext.Provider value={value}>{children}</LiteRTContext.Provider>
  );
}
