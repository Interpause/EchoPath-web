import { createContext, useContext } from "react";

export interface LiteRTContextState {
  model: null;
  isLoading: boolean;
  error: unknown;
}

export const LiteRTContext = createContext<LiteRTContextState | null>(null);

export function useLiteRT() {
  const context = useContext(LiteRTContext);
  if (!context) {
    throw new Error("useLiteRT must be used within a LiteRTProvider");
  }
  return context;
}
