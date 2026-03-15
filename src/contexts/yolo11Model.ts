import * as ort from "onnxruntime-web/webgpu";
import type { BBox, InferYOLO11Options, PixelSource } from "./OnnxContext";

const YOLO11_INPUT_SIZE = 640;
const YOLO11_BOX_CHANNELS = 4;
const DEFAULT_CONFIDENCE_THRESHOLD = 0.25;
const DEFAULT_IOU_THRESHOLD = 0.45;
const DEFAULT_MAX_DETECTIONS = 100;
const MODEL_CANDIDATES = ["/models/yolo11n.onnx"];

function getSourceDimensions(source: PixelSource) {
  if (source instanceof HTMLImageElement) {
    return {
      width: source.naturalWidth || source.width,
      height: source.naturalHeight || source.height,
    };
  }

  return { width: source.width, height: source.height };
}

function preprocessYOLO11Input(
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

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function computeIntersectionOverUnion(a: BBox, b: BBox) {
  const overlapWidth = Math.max(
    0,
    Math.min(a.xMax, b.xMax) - Math.max(a.xMin, b.xMin),
  );
  const overlapHeight = Math.max(
    0,
    Math.min(a.yMax, b.yMax) - Math.max(a.yMin, b.yMin),
  );
  const intersection = overlapWidth * overlapHeight;
  if (intersection <= 0) {
    return 0;
  }

  const areaA = Math.max(0, a.xMax - a.xMin) * Math.max(0, a.yMax - a.yMin);
  const areaB = Math.max(0, b.xMax - b.xMin) * Math.max(0, b.yMax - b.yMin);
  const union = areaA + areaB - intersection;

  if (union <= 0) {
    return 0;
  }

  return intersection / union;
}

function applyNonMaximumSuppression(
  boxes: BBox[],
  iouThreshold: number,
  maxDetections: number,
): BBox[] {
  const sortedBoxes = [...boxes].sort((a, b) => b.score - a.score);
  const selectedBoxes: BBox[] = [];

  for (const candidate of sortedBoxes) {
    if (selectedBoxes.length >= maxDetections) {
      break;
    }

    const overlapsExistingBox = selectedBoxes.some(
      (selected) =>
        selected.classIndex === candidate.classIndex &&
        computeIntersectionOverUnion(selected, candidate) > iouThreshold,
    );

    if (!overlapsExistingBox) {
      selectedBoxes.push(candidate);
    }
  }

  return selectedBoxes;
}

function getOutputShape(outputTensor: ort.Tensor) {
  const dims = outputTensor.dims;

  if (dims.length === 3 && dims[0] === 1) {
    return { channelCount: dims[1], predictionCount: dims[2] };
  }

  if (dims.length === 2) {
    return { channelCount: dims[0], predictionCount: dims[1] };
  }

  return null;
}

function decodeYOLO11Detections(
  outputTensor: ort.Tensor,
  inputSize: number,
  sourceWidth: number,
  sourceHeight: number,
  confidenceThreshold: number,
  iouThreshold: number,
  maxDetections: number,
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

  const outputShape = getOutputShape(outputTensor);
  if (!outputShape) {
    throw new Error(
      `Unsupported YOLO11 output dimensions: ${outputTensor.dims.join("x")}`,
    );
  }

  const { channelCount, predictionCount } = outputShape;
  if (channelCount <= YOLO11_BOX_CHANNELS || predictionCount === 0) {
    return [];
  }

  const classCount = channelCount - YOLO11_BOX_CHANNELS;
  const xScale = sourceWidth / inputSize;
  const yScale = sourceHeight / inputSize;
  const boxes: BBox[] = [];

  for (
    let predictionIndex = 0;
    predictionIndex < predictionCount;
    predictionIndex += 1
  ) {
    let bestScore = 0;
    let bestClassIndex = -1;

    for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
      const rawScore =
        floatData[
          (YOLO11_BOX_CHANNELS + classIndex) * predictionCount + predictionIndex
        ];
      const score = toProbability(rawScore);

      if (score > bestScore) {
        bestScore = score;
        bestClassIndex = classIndex;
      }
    }

    if (bestScore < confidenceThreshold || bestClassIndex < 0) {
      continue;
    }

    const centerX = floatData[predictionIndex];
    const centerY = floatData[predictionCount + predictionIndex];
    const width = floatData[2 * predictionCount + predictionIndex];
    const height = floatData[3 * predictionCount + predictionIndex];

    if (!(width > 0) || !(height > 0)) {
      continue;
    }

    const halfWidth = width / 2;
    const halfHeight = height / 2;
    const xMin = clamp((centerX - halfWidth) * xScale, 0, sourceWidth);
    const yMin = clamp((centerY - halfHeight) * yScale, 0, sourceHeight);
    const xMax = clamp((centerX + halfWidth) * xScale, 0, sourceWidth);
    const yMax = clamp((centerY + halfHeight) * yScale, 0, sourceHeight);

    if (xMax <= xMin || yMax <= yMin) {
      continue;
    }

    boxes.push({
      xMin,
      yMin,
      xMax,
      yMax,
      score: bestScore,
      classIndex: bestClassIndex,
    });
  }

  return applyNonMaximumSuppression(boxes, iouThreshold, maxDetections);
}

export async function loadYOLO11Model(): Promise<ort.InferenceSession> {
  const loadErrors: string[] = [];

  for (const modelPath of MODEL_CANDIDATES) {
    try {
      return await ort.InferenceSession.create(modelPath, {
        executionProviders: ["webgl", "wasm"],
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

export async function inferYOLO11Model(
  model: ort.InferenceSession,
  source: PixelSource,
  options: InferYOLO11Options = {},
): Promise<BBox[]> {
  const {
    inputSize = YOLO11_INPUT_SIZE,
    confidenceThreshold = DEFAULT_CONFIDENCE_THRESHOLD,
    normalizeInput = true,
    iouThreshold = DEFAULT_IOU_THRESHOLD,
    maxDetections = DEFAULT_MAX_DETECTIONS,
  } = options;

  const { width, height } = getSourceDimensions(source);
  const inputTensor = preprocessYOLO11Input(source, inputSize, normalizeInput);

  const inputName = model.inputNames[0];
  if (!inputName) {
    throw new Error("Model has no input tensor.");
  }

  const outputMap = await model.run({ [inputName]: inputTensor });
  const outputTensor = getFirstOutputTensor(outputMap);
  if (!outputTensor) {
    return [];
  }

  return decodeYOLO11Detections(
    outputTensor,
    inputSize,
    width,
    height,
    confidenceThreshold,
    iouThreshold,
    maxDetections,
  );
}
