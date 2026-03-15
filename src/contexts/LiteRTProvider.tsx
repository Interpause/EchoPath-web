import type { PropsWithChildren } from "react";
import { useEffect, useState, useRef } from "react";
import { LiteRTContext } from "./LiteRTContext";
import { loadLiteRt } from "@litertjs/core";

async function loadRuntime() {
  await loadLiteRt(`/litert/`);
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
