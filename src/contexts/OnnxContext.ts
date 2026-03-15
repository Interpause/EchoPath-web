import { createContext, useContext } from "react";
import type * as ort from "onnxruntime-web";

export type PixelSource =
  | ImageData
  | HTMLImageElement
  | HTMLCanvasElement
  | HTMLVideoElement
  | ImageBitmap;

export interface BBox {
  xMin: number;
  yMin: number;
  xMax: number;
  yMax: number;
  score: number;
  classIndex: number;
}

export interface InferYOLOOptions {
  inputSize?: number;
  confidenceThreshold?: number;
  normalizeInput?: boolean;
}

type InferYOLOModelFn = (
  source: PixelSource,
  options?: InferYOLOOptions,
) => Promise<BBox[]>;

export interface OnnxContextState {
  model: ort.InferenceSession | null;
  isLoading: boolean;
  error: unknown | null;
  inferYOLOModel: InferYOLOModelFn | null;
}

export const OnnxContext = createContext<OnnxContextState | null>(null);

export function useOnnx() {
  const context = useContext(OnnxContext);
  if (!context) {
    throw new Error("useOnnx must be used within a OnnxProvider");
  }
  return context;
}
