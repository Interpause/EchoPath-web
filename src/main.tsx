import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@/index.css";
import App from "@/App";
import { OnnxProvider } from "@/contexts/OnnxProvider";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <OnnxProvider>
      <App />
    </OnnxProvider>
  </StrictMode>,
);
