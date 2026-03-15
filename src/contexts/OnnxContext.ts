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

export interface InferYOLO26Options {
  inputSize?: number;
  confidenceThreshold?: number;
  normalizeInput?: boolean;
}

export interface InferYOLO11Options extends InferYOLO26Options {
  iouThreshold?: number;
  maxDetections?: number;
}

export type InferObjDetOptions = InferYOLO11Options;

type InferObjDetModel = (
  source: PixelSource,
  options?: InferObjDetOptions,
) => Promise<BBox[]>;

export interface OnnxContextState {
  model: ort.InferenceSession | null;
  isLoading: boolean;
  error: unknown | null;
  inferObjDetModel: InferObjDetModel | null;
}

export const OnnxContext = createContext<OnnxContextState | null>(null);

export function useOnnx() {
  const context = useContext(OnnxContext);
  if (!context) {
    throw new Error("useOnnx must be used within a OnnxProvider");
  }
  return context;
}
