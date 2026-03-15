import type { PropsWithChildren } from "react";
import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";
import { OnnxContext } from "./OnnxContext";
import type { BBox, InferYOLOOptions, PixelSource } from "./OnnxContext";

const YOLO_INPUT_SIZE = 640;
const YOLO_OUTPUT_STRIDE = 6;
const DEFAULT_CONFIDENCE_THRESHOLD = 0.1;
const MODEL_CANDIDATES = ["/models/yolo26n.onnx"];

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
): ort.Tensor {
  const canvas = document.createElement("canvas");
  canvas.width = inputSize;
  canvas.height = inputSize;

  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    throw new Error("Failed to create canvas context for preprocessing.");
  }

  if (source instanceof ImageData) {
    const sourceCanvas = document.createElement("canvas");
    sourceCanvas.width = source.width;
    sourceCanvas.height = source.height;
    const sourceContext = sourceCanvas.getContext("2d", {
      willReadFrequently: true,
    });
    if (!sourceContext) {
      throw new Error("Failed to create canvas context for ImageData input.");
    }

    sourceContext.putImageData(source, 0, 0);
    context.drawImage(sourceCanvas, 0, 0, inputSize, inputSize);
  } else {
    context.drawImage(source, 0, 0, inputSize, inputSize);
  }

  const { data } = context.getImageData(0, 0, inputSize, inputSize);
  const pixelCount = inputSize * inputSize;
  const tensorData = new Float32Array(3 * pixelCount);
  const scale = normalizeInput ? 1 / 255 : 1;

  for (let i = 0; i < pixelCount; i += 1) {
    const dataOffset = i * 4;
    tensorData[i] = data[dataOffset] * scale;
    tensorData[pixelCount + i] = data[dataOffset + 1] * scale;
    tensorData[2 * pixelCount + i] = data[dataOffset + 2] * scale;
  }

  return new ort.Tensor("float32", tensorData, [1, 3, inputSize, inputSize]);
}

function getFirstOutputTensor(outputMap: ort.InferenceSession.ReturnType) {
  const firstOutputName = Object.keys(outputMap)[0];
  if (!firstOutputName) {
    return null;
  }
  return outputMap[firstOutputName] ?? null;
}

function toProbability(rawScore: number) {
  if (!Number.isFinite(rawScore)) {
    return 0;
  }

  if (rawScore >= 0 && rawScore <= 1) {
    return rawScore;
  }

  return 1 / (1 + Math.exp(-rawScore));
}

function decodeDetections(
  outputTensor: ort.Tensor,
  inputSize: number,
  sourceWidth: number,
  sourceHeight: number,
  confidenceThreshold: number,
): BBox[] {
  const outputData = outputTensor.data;
  const floatData =
    outputData instanceof Float32Array
      ? outputData
      : outputData instanceof Float64Array
        ? Float32Array.from(outputData)
        : null;

  if (!floatData) {
    return [];
  }

  const maxDetections = Math.floor(floatData.length / YOLO_OUTPUT_STRIDE);
  if (maxDetections === 0) {
    return [];
  }

  const boxes: BBox[] = [];

  for (let i = 0; i < maxDetections; i += 1) {
    const offset = i * YOLO_OUTPUT_STRIDE;
    const score = toProbability(floatData[offset + 4]);
    if (score < confidenceThreshold) {
      continue;
    }

    const classIndex = Math.round(floatData[offset + 5]);
    let xMin = floatData[offset + 0];
    let yMin = floatData[offset + 1];
    let xMax = floatData[offset + 2];
    let yMax = floatData[offset + 3];

    xMin *= sourceWidth / inputSize;
    xMax *= sourceWidth / inputSize;
    yMin *= sourceHeight / inputSize;
    yMax *= sourceHeight / inputSize;

    boxes.push({
      xMin: Math.max(0, xMin),
      yMin: Math.max(0, yMin),
      xMax: Math.min(sourceWidth, xMax),
      yMax: Math.min(sourceHeight, yMax),
      score,
      classIndex,
    });
  }

  return boxes.sort((a, b) => b.score - a.score);
}

async function loadYOLOModel(): Promise<ort.InferenceSession> {
  const loadErrors: string[] = [];

  for (const modelPath of MODEL_CANDIDATES) {
    try {
      return await ort.InferenceSession.create(modelPath, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      loadErrors.push(`${modelPath}: ${message}`);
    }
  }

  throw new Error(
    `Unable to load ONNX model. Tried ${MODEL_CANDIDATES.join(", ")}. Details: ${loadErrors.join(" | ")}`,
  );
}

async function inferYOLOModel(
  model: ort.InferenceSession,
  source: PixelSource,
  options: InferYOLOOptions = {},
): Promise<BBox[]> {
  const {
    inputSize = YOLO_INPUT_SIZE,
    confidenceThreshold = DEFAULT_CONFIDENCE_THRESHOLD,
    normalizeInput = true,
  } = options;

  const { width, height } = getSourceDimensions(source);
  const inputTensor = preprocessYOLOInput(source, inputSize, normalizeInput);

  const inputName = model.inputNames[0];
  if (!inputName) {
    throw new Error("Model has no input tensor.");
  }

  const outputMap = await model.run({ [inputName]: inputTensor });
  const outputTensor = getFirstOutputTensor(outputMap);
  if (!outputTensor) {
    return [];
  }

  return decodeDetections(
    outputTensor,
    inputSize,
    width,
    height,
    confidenceThreshold,
  );
}

export function OnnxProvider({ children }: PropsWithChildren) {
  const [model, setModel] = useState<ort.InferenceSession | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<unknown | null>(null);

  const infer = async (source: PixelSource, options?: InferYOLOOptions) => {
    if (!model) {
      return [];
    }

    return inferYOLOModel(model, source, options);
  };

  const isInitializing = useRef(false);

  useEffect(() => {
    if (isInitializing.current) return;
    isInitializing.current = true;

    async function init() {
      try {
        const compiledModel = await loadYOLOModel();
        setModel(compiledModel);
      } catch (err) {
        console.error(err);
        setError(err);
      } finally {
        setIsLoading(false);
      }
    }

    init();
  }, []);

  const value = {
    model,
    isLoading,
    error,
    inferYOLOModel: model ? infer : null,
  };

  return <OnnxContext.Provider value={value}>{children}</OnnxContext.Provider>;
}
