import type { PropsWithChildren } from "react";
import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web/webgl";
import { OnnxContext } from "./OnnxContext";
import type { InferObjDetOptions, PixelSource } from "./OnnxContext";
import { inferYOLO26Model, loadYOLO26Model } from "./yolo26Model";
import { inferYOLO11Model, loadYOLO11Model } from "./yolo11Model";

const modelImplementations = {
  yolo26: {
    load: loadYOLO26Model,
    infer: (
      model: ort.InferenceSession,
      source: PixelSource,
      options?: InferObjDetOptions,
    ) => inferYOLO26Model(model, source, options),
  },
  yolo11: {
    load: loadYOLO11Model,
    infer: (
      model: ort.InferenceSession,
      source: PixelSource,
      options?: InferObjDetOptions,
    ) => inferYOLO11Model(model, source, options),
  },
};

// const activeModelKey = "yolo26";
const activeModelKey = "yolo11";

const activeModel = modelImplementations[activeModelKey];

export function OnnxProvider({ children }: PropsWithChildren) {
  const [model, setModel] = useState<ort.InferenceSession | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<unknown | null>(null);

  const infer = async (source: PixelSource, options?: InferObjDetOptions) => {
    if (!model) {
      return [];
    }

    return activeModel.infer(model, source, options);
  };

  const isInitializing = useRef(false);

  useEffect(() => {
    if (isInitializing.current) return;
    isInitializing.current = true;

    async function init() {
      try {
        const compiledModel = await activeModel.load();
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
    inferObjDetModel: model ? infer : null,
  };

  return <OnnxContext.Provider value={value}>{children}</OnnxContext.Provider>;
}
