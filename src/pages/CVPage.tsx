import { useEffect, useMemo, useState } from "react";
import { CameraView } from "capacitor-camera-view";
import { useLiteRT } from "@/contexts/LiteRTContext";
import { DetectionOverlay } from "@/components/DetectionOverlay";
import type { BBox } from "@/contexts/LiteRTContext";
import "./CVPage.css";

interface SnapshotState {
  src: string;
  width: number;
  height: number;
}

function toDataUrl(base64: string) {
  if (base64.startsWith("data:")) {
    return base64;
  }

  return `data:image/jpeg;base64,${base64}`;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Failed to decode captured frame."));
    image.src = src;
  });
}

export function CVPage() {
  const { isLoading, error, inferYOLOModel } = useLiteRT();
  const [isCameraRunning, setIsCameraRunning] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [detections, setDetections] = useState<BBox[]>([]);
  const [snapshot, setSnapshot] = useState<SnapshotState | null>(null);

  const isInferenceReady = !isLoading && !error && !!inferYOLOModel;

  const statusText = useMemo(() => {
    if (isLoading) return "Loading LiteRT + YOLO model…";
    if (error) return "Failed to load model runtime.";
    if (!inferYOLOModel) return "Model unavailable.";
    return "Model ready.";
  }, [error, inferYOLOModel, isLoading]);

  const startCamera = async () => {
    try {
      setCameraError(null);

      const permission = await CameraView.checkPermissions();
      if (permission.camera !== "granted") {
        const requested = await CameraView.requestPermissions();
        if (requested.camera !== "granted") {
          throw new Error("Camera permission was not granted.");
        }
      }

      await CameraView.start({
        position: "back",
        containerElementId: "cameraContainer",
      });

      document.body.classList.add("camera-running");
      setIsCameraRunning(true);
    } catch (err) {
      document.body.classList.remove("camera-running");
      setIsCameraRunning(false);
      setCameraError(
        err instanceof Error ? err.message : "Failed to start camera.",
      );
    }
  };

  const stopCamera = async () => {
    try {
      document.body.classList.remove("camera-running");
      await CameraView.stop();
      setIsCameraRunning(false);
    } catch (err) {
      setCameraError(
        err instanceof Error ? err.message : "Failed to stop camera.",
      );
    }
  };

  const captureAndInfer = async () => {
    if (!inferYOLOModel || !isCameraRunning) return;

    setIsCapturing(true);
    setCameraError(null);

    try {
      const capture = await CameraView.captureSample({ quality: 80 });
      const src = toDataUrl(capture.photo);
      const image = await loadImage(src);
      const boxes = await inferYOLOModel(image, {
        confidenceThreshold: 0.1,
        normalizeInput: false,
      });

      setSnapshot({
        src,
        width: image.naturalWidth,
        height: image.naturalHeight,
      });
      setDetections(boxes);
    } catch (err) {
      setCameraError(
        err instanceof Error ? err.message : "Failed to capture and infer.",
      );
    } finally {
      setIsCapturing(false);
    }
  };

  useEffect(() => {
    return () => {
      document.body.classList.remove("camera-running");
      CameraView.stop().catch(() => {
        // noop - app might already be stopped/unmounted
      });
    };
  }, []);

  return (
    <main className="cv-page camera-modal">
      <header className="cv-header">
        <h1>Camera + YOLO</h1>
        <p>{statusText}</p>
      </header>

      <section className="cv-controls" aria-label="Camera controls">
        <button onClick={startCamera} disabled={isCameraRunning || isLoading}>
          Start Camera
        </button>
        <button onClick={stopCamera} disabled={!isCameraRunning}>
          Stop Camera
        </button>
        <button
          onClick={captureAndInfer}
          disabled={!isCameraRunning || !isInferenceReady || isCapturing}
        >
          {isCapturing ? "Running Inference…" : "Capture + Infer"}
        </button>
      </section>

      {cameraError ? <p className="cv-error">{cameraError}</p> : null}

      <section className="cv-layout">
        <div className="cv-camera-panel">
          <h2>Camera Preview</h2>
          <div id="cameraContainer" className="camera-container" />
        </div>

        <div className="cv-result-panel">
          <h2>Last Captured Frame</h2>
          {snapshot ? (
            <div className="snapshot-wrapper">
              <img
                src={snapshot.src}
                alt="Captured camera frame"
                className="snapshot"
              />
              <DetectionOverlay
                boxes={detections}
                imageWidth={snapshot.width}
                imageHeight={snapshot.height}
              />
            </div>
          ) : (
            <p className="cv-empty">Capture a frame to run inference.</p>
          )}

          <div className="detection-list">
            <h3>Detections ({detections.length})</h3>
            {detections.length === 0 ? (
              <p className="cv-empty">No detections yet.</p>
            ) : (
              <ul>
                {detections.map((box, index) => (
                  <li key={`${box.classIndex}-${box.score}-${index}`}>
                    {`#${index + 1} · class ${box.classIndex} · ${(box.score * 100).toFixed(1)}%`}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}
