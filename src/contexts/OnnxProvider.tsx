import type { PropsWithChildren } from "react";
import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web/webgpu";
import { OnnxContext } from "./OnnxContext";
import type { InferYOLO26Options, PixelSource } from "./OnnxContext";
import { inferYOLO26Model, loadYOLO26Model } from "./yolo26Model";

export function OnnxProvider({ children }: PropsWithChildren) {
  const [model, setModel] = useState<ort.InferenceSession | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<unknown | null>(null);

  const infer = async (source: PixelSource, options?: InferYOLO26Options) => {
    if (!model) {
      return [];
    }

    return inferYOLO26Model(model, source, options);
  };

  const isInitializing = useRef(false);

  useEffect(() => {
    if (isInitializing.current) return;
    isInitializing.current = true;

    async function init() {
      try {
        const compiledModel = await loadYOLO26Model();
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
    inferYOLO26Model: model ? infer : null,
  };

  return <OnnxContext.Provider value={value}>{children}</OnnxContext.Provider>;
}
