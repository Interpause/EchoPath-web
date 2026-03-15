import { useCallback, useEffect, useRef, useState } from "react";
import { CameraView } from "capacitor-camera-view";
import { useOnnx } from "@/contexts/OnnxContext";
import { DetectionOverlay } from "@/components/DetectionOverlay";
import type { BBox } from "@/contexts/OnnxContext";
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
  const { isLoading, error, inferObjDetModel: inferYOLO26Model } = useOnnx();
  const [isCameraRunning, setIsCameraRunning] = useState(false);
  const [isInferring, setIsInferring] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [detections, setDetections] = useState<BBox[]>([]);
  const [snapshot, setSnapshot] = useState<SnapshotState | null>(null);
  const [fps, setFps] = useState<number | null>(null);

  // Refs so the rAF loop never captures stale closures
  const loopActiveRef = useRef(false);
  const inferInProgressRef = useRef(false);
  const inferFnRef = useRef(inferYOLO26Model);
  const cameraRunningRef = useRef(isCameraRunning);
  const lastCompleteTimeRef = useRef<number>(0);

  useEffect(() => {
    inferFnRef.current = inferYOLO26Model;
  }, [inferYOLO26Model]);

  useEffect(() => {
    cameraRunningRef.current = isCameraRunning;
  }, [isCameraRunning]);

  const isInferenceReady = !isLoading && !error && !!inferYOLO26Model;

  const statusText = (() => {
    if (isLoading) return "Loading ONNX Runtime + YOLO26 model…";
    if (error) return "Failed to load model runtime.";
    if (!inferYOLO26Model) return "Model unavailable.";
    return "Model ready.";
  })();

  // rAF-driven inference loop — drops frames if previous inference is still running
  const runLoop = useCallback(() => {
    if (!loopActiveRef.current) return;

    const infer = inferFnRef.current;
    if (!inferInProgressRef.current && infer && cameraRunningRef.current) {
      inferInProgressRef.current = true;
      CameraView.captureSample({ quality: 80 })
        .then((capture) => {
          const src = toDataUrl(capture.photo);
          return loadImage(src).then((image) => ({ src, image }));
        })
        .then(({ src, image }) =>
          infer(image, { confidenceThreshold: 0.1 }).then((boxes) => ({
            src,
            image,
            boxes,
          })),
        )
        .then(({ src, image, boxes }) => {
          if (!loopActiveRef.current) return; // stopped while in-flight — discard
          const now = performance.now();
          if (lastCompleteTimeRef.current > 0) {
            setFps(1000 / (now - lastCompleteTimeRef.current));
          }
          lastCompleteTimeRef.current = now;
          setSnapshot({
            src,
            width: image.naturalWidth,
            height: image.naturalHeight,
          });
          setDetections(boxes);
        })
        .catch((err) => {
          if (loopActiveRef.current) {
            setCameraError(
              err instanceof Error ? err.message : "Inference error.",
            );
          }
        })
        .finally(() => {
          inferInProgressRef.current = false;
        });
    }

    requestAnimationFrame(runLoop);
  }, []);

  const startInference = () => {
    if (!inferYOLO26Model || !isCameraRunning) return;
    setCameraError(null);
    lastCompleteTimeRef.current = 0;
    setFps(null);
    loopActiveRef.current = true;
    setIsInferring(true);
    requestAnimationFrame(runLoop);
  };

  const stopInference = () => {
    loopActiveRef.current = false;
    setIsInferring(false);
  };

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
    stopInference();
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

  useEffect(() => {
    return () => {
      loopActiveRef.current = false;
      document.body.classList.remove("camera-running");
      CameraView.stop().catch(() => {
        // noop - app might already be stopped/unmounted
      });
    };
  }, []);

  return (
    <main className="cv-page camera-modal">
      <header className="cv-header">
        <h1>Camera + YOLO26</h1>
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
          onClick={isInferring ? stopInference : startInference}
          disabled={!isCameraRunning || !isInferenceReady}
        >
          {isInferring ? "Stop Inference" : "Start Inference"}
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
            <p className="cv-empty">Start inference to see captures.</p>
          )}

          <div className="detection-list">
            <div className="detection-list-header">
              <h3>Detections ({detections.length})</h3>
              {fps !== null ? (
                <span className="fps-display">{fps.toFixed(1)} FPS</span>
              ) : null}
            </div>
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
