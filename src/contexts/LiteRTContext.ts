import { createContext, useContext } from "react";
import type { CompiledModel } from "@litertjs/core";

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

export interface LiteRTContextState {
  model: CompiledModel | null;
  isLoading: boolean;
  error: unknown | null;
  inferYOLOModel: InferYOLOModelFn | null;
}

export const LiteRTContext = createContext<LiteRTContextState | null>(null);

export function useLiteRT() {
  const context = useContext(LiteRTContext);
  if (!context) {
    throw new Error("useLiteRT must be used within a LiteRTProvider");
  }
  return context;
}
