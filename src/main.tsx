import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@/index.css";
import App from "@/App";
import { LiteRTProvider } from "@/contexts/LiteRTProvider";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <LiteRTProvider>
      <App />
    </LiteRTProvider>
  </StrictMode>,
);
