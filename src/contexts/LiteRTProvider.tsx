import type { PropsWithChildren } from "react";
import { useEffect, useState, useRef } from "react";
import { LiteRTContext } from "./LiteRTContext";
import type { BBox, InferYOLOOptions, PixelSource } from "./LiteRTContext";
import type { CompiledModel } from "@litertjs/core";
import { loadAndCompile, loadLiteRt, getWebGpuDevice } from "@litertjs/core";
import { runWithTfjsTensors } from "@litertjs/tfjs-interop";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu";
import { WebGPUBackend } from "@tensorflow/tfjs-backend-webgpu";

const YOLO_INPUT_SIZE = 640;
const YOLO_OUTPUT_STRIDE = 6;
const DEFAULT_CONFIDENCE_THRESHOLD = 0.1;

function getSourceDimensions(source: PixelSource) {
  if (source instanceof HTMLImageElement) {
    return {
      width: source.naturalWidth || source.width,
      height: source.naturalHeight || source.height,
    };
  }

  return { width: source.width, height: source.height };
}

function preprocessYOLOInput(
  source: PixelSource,
  inputSize: number,
  normalizeInput: boolean,
): tf.Tensor4D {
  return tf.tidy(() => {
    let tensor = tf.browser.fromPixels(source, 3).toFloat();
    tensor = tf.image.resizeBilinear(tensor, [inputSize, inputSize]);

    if (normalizeInput) {
      tensor = tensor.div(255);
    }

    return tensor.expandDims(0) as tf.Tensor4D;
  });
}

async function loadRuntime() {
  // Initialize TensorFlow.js WebGPU backend
  await tf.setBackend("webgpu");

  // Initialize LiteRT.js's Wasm files
  await loadLiteRt(`/litert/`);

  // Make TFJS use the same GPU device as LiteRT.js (for tensor conversion)
  const device = await getWebGpuDevice();
  tf.removeBackend("webgpu");
  if (!device) {
    console.error("No WebGPU device available, falling back to CPU backend...");
    await tf.setBackend("cpu");
    return;
  }
  tf.registerBackend("webgpu", () => new WebGPUBackend(device));
  await tf.setBackend("webgpu");
}

async function loadYOLOModel(): Promise<CompiledModel> {
  const hasWebGpuDevice = !!(await getWebGpuDevice());

  return loadAndCompile("/models/yolo26n_float16.tflite", {
    accelerator: hasWebGpuDevice ? "webgpu" : "wasm",
  });
}

async function inferYOLOModel(
  model: CompiledModel,
  source: PixelSource,
  options: InferYOLOOptions = {},
): Promise<BBox[]> {
  const {
    inputSize = YOLO_INPUT_SIZE,
    confidenceThreshold = DEFAULT_CONFIDENCE_THRESHOLD,
    normalizeInput = false,
  } = options;

  const { width, height } = getSourceDimensions(source);
  const imageTensor = preprocessYOLOInput(source, inputSize, normalizeInput);
  let outputs: tf.Tensor[] = [];

  try {
    outputs = (await runWithTfjsTensors(model, [imageTensor])) as tf.Tensor[];
    if (outputs.length === 0) {
      return [];
    }

    const outputValues = await outputs[0].data();
    const maxDetections = Math.floor(outputValues.length / YOLO_OUTPUT_STRIDE);
    const boxes: BBox[] = [];

    for (let i = 0; i < maxDetections; i++) {
      const offset = i * YOLO_OUTPUT_STRIDE;
      const score = outputValues[offset + 4];

      if (!Number.isFinite(score) || score < confidenceThreshold) {
        continue;
      }

      const classIndex = Math.round(outputValues[offset + 5]);
      const xMin = outputValues[offset + 0] * width;
      const yMin = outputValues[offset + 1] * height;
      const xMax = outputValues[offset + 2] * width;
      const yMax = outputValues[offset + 3] * height;

      boxes.push({
        xMin,
        yMin,
        xMax,
        yMax,
        score,
        classIndex,
      });
    }

    return boxes;
  } finally {
    tf.dispose(imageTensor);
    tf.dispose(outputs);
  }
}

export function LiteRTProvider({ children }: PropsWithChildren) {
  const [model, setModel] = useState<CompiledModel | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<unknown | null>(null);

  const infer = async (source: PixelSource, options?: InferYOLOOptions) => {
    if (!model) {
      return [];
    }

    return inferYOLOModel(model, source, options);
  };

  // Use a ref to prevent double-initialization in React Strict Mode
  const isInitializing = useRef(false);

  useEffect(() => {
    if (isInitializing.current) return;
    isInitializing.current = true;

    async function init() {
      try {
        await loadRuntime();
        const compiledModel = await loadYOLOModel();
        setModel(compiledModel);
      } catch (err) {
        setError(err);
      } finally {
        setIsLoading(false);
      }
    }

    init();
  }, []);

  // 3. Memoize the value so consumers don't re-render unnecessarily
  const value = {
    model,
    isLoading,
    error,
    inferYOLOModel: model ? infer : null,
  };

  return (
    <LiteRTContext.Provider value={value}>{children}</LiteRTContext.Provider>
  );
}
