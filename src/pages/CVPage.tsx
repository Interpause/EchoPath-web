import { useEffect, useRef, useState } from "react";
import { CameraView } from "capacitor-camera-view";
import { SpatialAudioEngine } from "../../lib/sound";
import "./CVPage.css";

const DEFAULT_FRAME_SEND_INTERVAL_MS = 250;
const DEFAULT_AUDIO_MIN_INTERVAL_MS = 350;
const DEFAULT_BACKEND_WS_ENDPOINT = "/ws";
const WAKE_WORD = "hey john";
const COMMAND_LISTEN_WINDOW_MS = 7000;
const SPEECH_RETRY_DELAY_MS = 1200;
const SPEECH_FATAL_ERRORS = new Set([
  "not-allowed",
  "service-not-allowed",
  "audio-capture",
  "language-not-supported",
]);

type SpeechRecognitionAlternative = {
  transcript: string;
};

type SpeechRecognitionResultLike = {
  isFinal: boolean;
  length?: number;
  0?: SpeechRecognitionAlternative;
  [index: number]: SpeechRecognitionAlternative | number | boolean | undefined;
};

type SpeechRecognitionEventLike = Event & {
  results: ArrayLike<SpeechRecognitionResultLike>;
};

interface SpeechRecognitionLike extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  onstart: ((event: Event) => void) | null;
  onend: ((event: Event) => void) | null;
  onerror: ((event: Event & { error?: string }) => void) | null;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  start: () => void;
  stop: () => void;
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

function getSpeechRecognitionConstructor() {
  const speechWindow = window as Window & {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  };

  return speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition ?? null;
}

function toWebSocketUrl(rawUrl: string): string {
  if (rawUrl.startsWith("ws://") || rawUrl.startsWith("wss://")) {
    return rawUrl;
  }

  if (rawUrl.startsWith("http://")) {
    return `ws://${rawUrl.slice("http://".length)}`;
  }

  if (rawUrl.startsWith("https://")) {
    return `wss://${rawUrl.slice("https://".length)}`;
  }

  return rawUrl;
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error);
    reader.onloadend = () => {
      const result = reader.result;
      if (typeof result === "string") {
        // strip the data URL prefix if present
        resolve(result.replace(/^data:image\/[a-zA-Z]+;base64,/, ""));
      } else {
        reject(new Error("Failed to convert blob to base64."));
      }
    };
    reader.readAsDataURL(blob);
  });
}

type Detection = {
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
  label?: string;
  confidence?: number;
  dist_point?: DistPoint;
};

type DistPoint = {
  x: number;
  y: number;
  distance: number;
  label?: string;
  confidence?: number;
  z?: number;
};

type SpatialPoint = {
  x: number;
  y: number;
  z: number;
};

export function CVPage() {
  const [isCameraRunning, setIsCameraRunning] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [voiceStatus, setVoiceStatus] = useState("Voice idle.");
  const [lastSentAt, setLastSentAt] = useState<number | null>(null);
  const [captureFps, setCaptureFps] = useState(4);
  const [latestDetections, setLatestDetections] = useState<Detection[]>([]);
  const [latestDistPoints, setLatestDistPoints] = useState<DistPoint[]>([]);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const captureIntervalRef = useRef<number | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const audioEngineRef = useRef<SpatialAudioEngine | null>(null);
  const lastAudioAtRef = useRef(0);
  const isCameraRunningRef = useRef(false);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const keepRecognitionRunningRef = useRef(false);
  const waitingForCommandRef = useRef(false);
  const commandWindowDeadlineRef = useRef<number>(0);
  const commandWindowTimeoutRef = useRef<number | null>(null);
  const recognitionRestartTimeoutRef = useRef<number | null>(null);
  const lastTranscriptRef = useRef("");
  const lastTranscriptAtRef = useRef(0);

  const backendWsUrl = toWebSocketUrl(
    ((import.meta.env as any).VITE_BACKEND_WS_URL as string | undefined) ??
      DEFAULT_BACKEND_WS_ENDPOINT,
  );

  const statusText = isCameraRunning
    ? "Camera is running." +
      (lastSentAt ? ` Last frame sent: ${new Date(lastSentAt).toLocaleTimeString()}` : "")
    : "Camera is stopped.";

  const getDistPointsToRender = () => {
    const pointsFromDetections = latestDetections
      .map((d) => d?.dist_point)
      .filter(
        (p): p is DistPoint =>
          !!p &&
          Number.isFinite(p.x) &&
          Number.isFinite(p.y) &&
          Number.isFinite(p.distance),
      );

    if (pointsFromDetections.length > 0) {
      return pointsFromDetections;
    }

    return latestDistPoints.filter(
      (p): p is DistPoint =>
        !!p &&
        Number.isFinite(p.x) &&
        Number.isFinite(p.y) &&
        Number.isFinite(p.distance),
    );
  };

  const drawResults = () => {
    const overlay = overlayRef.current;
    const video = videoRef.current;
    if (!overlay || !video) return;

    const ctx = overlay.getContext("2d");
    if (!ctx) return;

    const width = Math.max(1, Math.floor(overlay.clientWidth));
    const height = Math.max(1, Math.floor(overlay.clientHeight));
    overlay.width = width;
    overlay.height = height;

    ctx.clearRect(0, 0, width, height);
    ctx.lineWidth = 2;
    ctx.font = "14px system-ui, sans-serif";

    latestDetections.forEach((detection) => {
      const x = detection.xmin * width;
      const y = detection.ymin * height;
      const boxWidth = (detection.xmax - detection.xmin) * width;
      const boxHeight = (detection.ymax - detection.ymin) * height;

      ctx.strokeStyle = "#22c55e";
      ctx.strokeRect(x, y, boxWidth, boxHeight);

      const label = detection.label ?? "obj";
      const conf =
        typeof detection.confidence === "number"
          ? detection.confidence.toFixed(2)
          : "";
      const text = conf ? `${label} ${conf}` : String(label);

      const pad = 4;
      const tw = ctx.measureText(text).width;
      const th = 16;
      ctx.fillStyle = "#22c55e";
      ctx.fillRect(x, Math.max(0, y - th), tw + pad * 2, th);
      ctx.fillStyle = "#111";
      ctx.fillText(text, x + pad, Math.max(12, y - 4));
    });

    getDistPointsToRender().forEach((point) => {
      const nx = Math.max(0, Math.min(1, Number(point.x)));
      const ny = Math.max(0, Math.min(1, Number(point.y)));
      const px = nx * width;
      const py = ny * height;

      ctx.fillStyle = "#f97316";
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fill();

      const distanceText = Number(point.distance).toFixed(2);
      const cls = point.label ?? "obj";
      const pointText = `(${distanceText}, ${cls})`;

      const pad = 4;
      const tw = ctx.measureText(pointText).width;
      const th = 16;
      const tx = Math.max(0, Math.min(px + 8, width - (tw + pad * 2)));
      const ty = Math.max(th, py - 8);

      ctx.fillStyle = "#f97316";
      ctx.fillRect(tx, ty - th, tw + pad * 2, th);
      ctx.fillStyle = "#111";
      ctx.fillText(pointText, tx + pad, ty - 4);
    });
  };

  const toSpatialPoint = (value: unknown): SpatialPoint | null => {
    if (!value || typeof value !== "object") {
      return null;
    }

    const candidate = value as Record<string, unknown>;
    const x = candidate.x;
    const y = candidate.y;
    const z = candidate.z;

    if (typeof x === "number" && typeof y === "number" && typeof z === "number") {
      return { x, y, z };
    }

    const nested = candidate.coordinates;
    if (nested && typeof nested === "object") {
      const coords = nested as Record<string, unknown>;
      const nx = coords.x;
      const ny = coords.y;
      const nz = coords.z;
      if (
        typeof nx === "number" &&
        typeof ny === "number" &&
        typeof nz === "number"
      ) {
        return { x: nx, y: ny, z: nz };
      }
    }

    return null;
  };

  const extractSpatialPoints = (message: unknown): SpatialPoint[] => {
    if (!message || typeof message !== "object") {
      return [];
    }

    const msg = message as Record<string, unknown>;
    const candidates: SpatialPoint[] = [];

    const directPoint = toSpatialPoint(msg);
    if (directPoint) {
      candidates.push(directPoint);
    }

    const arraysToCheck = [msg.points, msg.objects];
    const detections = msg.data;
    if (Array.isArray(detections)) {
      detections.forEach((item) => {
        if (!item || typeof item !== "object") {
          return;
        }

        const record = item as Record<string, unknown>;
        const distPoint = record.dist_point;
        if (!distPoint || typeof distPoint !== "object") {
          return;
        }

        const point = distPoint as Record<string, unknown>;
        const x = point.x;
        const y = point.y;
        const distance = point.distance;
        const z = point.z;

        if (
          typeof x === "number" &&
          typeof y === "number" &&
          typeof distance === "number"
        ) {
          candidates.push({
            x: (x - 0.5) * 2,
            y: (0.5 - y) * 2,
            z: typeof z === "number" ? z : -distance,
          });
        }
      });
    }

    const distPoints = msg.dist_points;
    if (Array.isArray(distPoints)) {
      distPoints.forEach((item) => {
        if (!item || typeof item !== "object") {
          return;
        }

        const point = item as Record<string, unknown>;
        const x = point.x;
        const y = point.y;
        const distance = point.distance;
        const z = point.z;

        if (
          typeof x === "number" &&
          typeof y === "number" &&
          typeof distance === "number"
        ) {
          candidates.push({
            x: (x - 0.5) * 2,
            y: (0.5 - y) * 2,
            z: typeof z === "number" ? z : -distance,
          });
        }
      });
    }

    const visualizer = msg.visualizer;
    if (visualizer && typeof visualizer === "object") {
      const v = visualizer as Record<string, unknown>;
      arraysToCheck.push(v.points, v.objects);
    }

    arraysToCheck.forEach((entry) => {
      if (!Array.isArray(entry)) {
        return;
      }
      entry.forEach((item) => {
        const point = toSpatialPoint(item);
        if (point) {
          candidates.push(point);
        }
      });
    });

    return candidates;
  };

  const ensureAudioEngine = () => {
    if (!audioEngineRef.current) {
      audioEngineRef.current = new SpatialAudioEngine({
        baseFrequency: 660,
        duration: 0.16,
        maxDistance: 30,
      });
    }

    return audioEngineRef.current;
  };

  const playSpatialCue = (points: SpatialPoint[]) => {
    if (!points.length) {
      return;
    }

    const now = Date.now();
    if (now - lastAudioAtRef.current < DEFAULT_AUDIO_MIN_INTERVAL_MS) {
      return;
    }

    const nearestPoint = points.reduce((best, point) => {
      const bestDist = Math.hypot(best.x, best.y, best.z);
      const pointDist = Math.hypot(point.x, point.y, point.z);
      return pointDist < bestDist ? point : best;
    });

    lastAudioAtRef.current = now;
    const engine = ensureAudioEngine();
    engine.playBeep(nearestPoint.x, nearestPoint.y, nearestPoint.z).catch(() => {
      // ignore individual audio failures
    });
  };

  const speakText = (text: string) => {
    if (!("speechSynthesis" in window)) {
      return;
    }

    const trimmed = text.trim();
    if (!trimmed) {
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(trimmed);
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  };

  const handleWebSocketMessage = (event: MessageEvent) => {
    try {
      const msg = JSON.parse(event.data) as Record<string, unknown>;

      if (msg.type === "detections" && Array.isArray(msg.data)) {
        setLatestDetections(msg.data as Detection[]);
        setLatestDistPoints(
          Array.isArray(msg.dist_points) ? (msg.dist_points as DistPoint[]) : [],
        );
      } else if (msg.type === "query_llm_response") {
        const responseText =
          typeof msg.data === "string" ? msg.data : "";
        if (responseText) {
          console.log("Received query_llm_response:", responseText);
          speakText(responseText);
          setVoiceStatus(`Assistant: ${responseText}`);
        }
      } else if (msg.type === "error") {
        const message =
          typeof msg.error === "string" ? msg.error : "Server error.";
        setBackendError(`Server error: ${message}`);
      }

      const spatialPoints = extractSpatialPoints(msg);
      playSpatialCue(spatialPoints);

      // Leave room for future messages (e.g., status, errors)
    } catch {
      // ignore malformed data
    }
  };

  const startWebSocket = () => {
    if (wsRef.current) return;

    const ws = new WebSocket(backendWsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.debug("WebSocket connected to", backendWsUrl);
    };

    ws.onmessage = handleWebSocketMessage;

    ws.onerror = (event) => {
      console.error("WebSocket error", event);
      setBackendError("WebSocket connection error.");
    };

    ws.onclose = () => {
      wsRef.current = null;
    };
  };

  const stopWebSocket = () => {
    const ws = wsRef.current;
    if (!ws) return;
    wsRef.current = null;
    ws.close();
  };

  const stopAudioEngine = () => {
    const engine = audioEngineRef.current;
    audioEngineRef.current = null;
    if (engine) {
      engine.dispose().catch(() => {
        // ignore cleanup errors
      });
    }
  };

  const sendFrameToBackend = async (imageBase64: string) => {
    try {
      setBackendError(null);
      const payload = {
        type: "image",
        data: imageBase64,
      };

      // Log JSON payload so it can be inspected in the browser devtools.
      console.log("Sending frame payload:", payload);

      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(payload));
      } else {
        setBackendError("WebSocket is not connected.");
        return;
      }

      setLastSentAt(Date.now());
    } catch (err) {
      setBackendError(
        err instanceof Error ? err.message : "Failed to send frame to backend.",
      );
    }
  };

  const sendQueryLlmToBackend = async (payload: {
    type: "query_llm";
    data: string;
    text: string;
  }) => {
    try {
      setBackendError(null);
      console.log("Sending query_llm payload:", payload);

      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(payload));
      } else {
        setBackendError("WebSocket is not connected.");
        return;
      }

      setLastSentAt(Date.now());
    } catch (err) {
      setBackendError(
        err instanceof Error
          ? err.message
          : "Failed to send query_llm payload to backend.",
      );
    }
  };

  const captureCurrentFrameBase64 = async (): Promise<string | null> => {
    const video = videoRef.current;
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return null;
    }

    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) {
      return null;
    }

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return null;
    }

    ctx.drawImage(video, 0, 0, width, height);
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.8),
    );

    if (!blob) {
      return null;
    }

    return blobToBase64(blob);
  };

  const handleVoiceCommand = async (command: string) => {
    const trimmed = command.trim();
    if (!trimmed) {
      setVoiceStatus("Heard wake word, but command was empty.");
      return;
    }

    const baseMessage = {
      type: "query_llm" as const,
      text: trimmed,
    };
    console.log("Wake command captured (query_llm):", baseMessage);

    const frameBase64 = await captureCurrentFrameBase64();
    if (!frameBase64) {
      console.log("query_llm not sent (no frame available):", baseMessage);
      setVoiceStatus("Command heard but no frame available yet.");
      return;
    }

    await sendQueryLlmToBackend({
      type: "query_llm",
      text:
        "In your response assume I am blind so do not use any kind of information that is unpercievable by me (colour, etc). Also BE AS BRIEF AND CONCISE as possible unless you are to instruct direction where you should be absolutely detailed and talk about what to touch and feel to direct a person who is blind. Use two sentences at most. The command follows: " +
        trimmed,
      data: frameBase64,
    });
    setVoiceStatus(`Sent command: \"${trimmed}\"`);
  };

  const clearCommandWindowTimeout = () => {
    if (commandWindowTimeoutRef.current !== null) {
      window.clearTimeout(commandWindowTimeoutRef.current);
      commandWindowTimeoutRef.current = null;
    }
  };

  const exitCommandWaiting = () => {
    waitingForCommandRef.current = false;
    commandWindowDeadlineRef.current = 0;
    clearCommandWindowTimeout();
  };

  const enterCommandWaiting = () => {
    waitingForCommandRef.current = true;
    commandWindowDeadlineRef.current = Date.now() + COMMAND_LISTEN_WINDOW_MS;
    clearCommandWindowTimeout();
    setVoiceStatus("Wake word detected. Waiting for command...");

    commandWindowTimeoutRef.current = window.setTimeout(() => {
      if (!waitingForCommandRef.current) {
        return;
      }

      waitingForCommandRef.current = false;
      commandWindowDeadlineRef.current = 0;
      commandWindowTimeoutRef.current = null;
      setVoiceStatus(`Listening for wake word \"${WAKE_WORD}\"...`);
    }, COMMAND_LISTEN_WINDOW_MS);
  };

  const scheduleRecognitionRestart = (
    recognition: SpeechRecognitionLike,
    reason: string,
  ) => {
    if (!keepRecognitionRunningRef.current || !isCameraRunningRef.current) {
      return;
    }

    if (recognitionRestartTimeoutRef.current !== null) {
      return;
    }

    recognitionRestartTimeoutRef.current = window.setTimeout(() => {
      recognitionRestartTimeoutRef.current = null;

      if (!keepRecognitionRunningRef.current || !isCameraRunningRef.current) {
        return;
      }

      try {
        recognition.start();
      } catch {
        // ignore transient restart failure
      }
    }, SPEECH_RETRY_DELAY_MS);

    setVoiceStatus(`${reason} Retrying...`);
  };

  const stopVoiceRecognition = () => {
    keepRecognitionRunningRef.current = false;
    exitCommandWaiting();
    if (recognitionRestartTimeoutRef.current !== null) {
      window.clearTimeout(recognitionRestartTimeoutRef.current);
      recognitionRestartTimeoutRef.current = null;
    }

    const recognition = recognitionRef.current;
    recognitionRef.current = null;

    if (recognition) {
      try {
        recognition.stop();
      } catch {
        // ignore stop failures
      }
    }

    setVoiceStatus("Voice idle.");
  };

  const startVoiceRecognition = () => {
    const Recognition = getSpeechRecognitionConstructor();
    if (!Recognition) {
      setVoiceStatus("Speech recognition unavailable on this device/browser.");
      return;
    }

    if (recognitionRef.current) {
      return;
    }

    const recognition = new Recognition();
    recognition.lang = "en-US";
    recognition.interimResults = true;
    recognition.continuous = true;
    recognition.maxAlternatives = 1;

    keepRecognitionRunningRef.current = true;
    exitCommandWaiting();
    if (recognitionRestartTimeoutRef.current !== null) {
      window.clearTimeout(recognitionRestartTimeoutRef.current);
      recognitionRestartTimeoutRef.current = null;
    }

    recognition.onstart = () => {
      setVoiceStatus(`Listening for wake word \"${WAKE_WORD}\"...`);
    };

    recognition.onend = () => {
      if (!keepRecognitionRunningRef.current || !isCameraRunningRef.current) {
        return;
      }

      scheduleRecognitionRestart(recognition, "Speech recognition ended.");
    };

    recognition.onerror = (event) => {
      const errorCode = event.error ?? "unknown";
      console.warn("Speech recognition error:", errorCode);

      if (SPEECH_FATAL_ERRORS.has(errorCode)) {
        keepRecognitionRunningRef.current = false;
        exitCommandWaiting();
        setVoiceStatus(`Speech recognition unavailable (${errorCode}).`);
        return;
      }

      scheduleRecognitionRestart(recognition, `Speech recognition error (${errorCode}).`);
    };

    recognition.onresult = (event) => {
      const resultCount = event.results.length;
      if (resultCount === 0) {
        return;
      }

      let transcript = "";
      for (let index = resultCount - 1; index >= 0; index -= 1) {
        const result = event.results[index];
        if (!result?.isFinal) {
          continue;
        }

        const candidate = result[0];
        const text =
          typeof candidate?.transcript === "string"
            ? candidate.transcript.trim()
            : "";

        if (text) {
          transcript = text;
          break;
        }
      }

      if (!transcript) {
        return;
      }

      const now = Date.now();
      const normalized = transcript.toLowerCase();

      if (
        normalized === lastTranscriptRef.current &&
        now - lastTranscriptAtRef.current < 1500
      ) {
        return;
      }

      lastTranscriptRef.current = normalized;
      lastTranscriptAtRef.current = now;

      const wakeIndex = normalized.indexOf(WAKE_WORD);

      if (waitingForCommandRef.current) {
        if (now > commandWindowDeadlineRef.current) {
          exitCommandWaiting();
          setVoiceStatus(`Listening for wake word \"${WAKE_WORD}\"...`);
          return;
        }

        if (wakeIndex !== -1) {
          const remainder = transcript.slice(wakeIndex + WAKE_WORD.length).trim();
          if (remainder) {
            exitCommandWaiting();
            void handleVoiceCommand(remainder);
            return;
          }

          enterCommandWaiting();
          return;
        }

        exitCommandWaiting();
        void handleVoiceCommand(transcript);
        return;
      }

      if (wakeIndex !== -1) {
        const remainder = transcript.slice(wakeIndex + WAKE_WORD.length).trim();
        if (remainder) {
          void handleVoiceCommand(remainder);
          return;
        }

        enterCommandWaiting();
      }
    };

    recognitionRef.current = recognition;

    try {
      recognition.start();
    } catch {
      setVoiceStatus("Unable to start speech recognition.");
    }
  };

  const captureFrame = async () => {
    const base64 = await captureCurrentFrameBase64();
    if (!base64) {
      return;
    }

    await sendFrameToBackend(base64);
  };

  const startFrameCaptureLoop = () => {
    if (captureIntervalRef.current != null) {
      return;
    }

    const fps = Math.max(1, Math.min(30, Number(captureFps) || 5));
    const interval = Math.floor(1000 / fps);

    captureIntervalRef.current = window.setInterval(
      () => {
        captureFrame().catch(() => {
          // ignore errors in individual frames so the loop continues
        });
      },
      Math.max(interval, DEFAULT_FRAME_SEND_INTERVAL_MS / 2),
    );
  };

  const stopFrameCaptureLoop = () => {
    if (captureIntervalRef.current != null) {
      window.clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
  };

  useEffect(() => {
    drawResults();
  }, [latestDetections, latestDistPoints]);

  useEffect(() => {
    drawResults();
    const onResize = () => drawResults();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
    };
  }, []);

  const stopMediaStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const startCamera = async () => {
    try {
      setCameraError(null);
      setBackendError(null);

      // Ensure permissions are granted where the Capacitor plugin is used.
      if (CameraView && CameraView.checkPermissions) {
        const permission = await CameraView.checkPermissions();
        if (permission.camera !== "granted") {
          const requested = await CameraView.requestPermissions();
          if (requested.camera !== "granted") {
            throw new Error("Camera permission was not granted.");
          }
        }
      }

      // Use the native camera view for permissions / platform support, but keep it hidden.
      await CameraView.start({
        position: "back",
        containerElementId: "cameraNativeContainer",
      });

      // Also start a regular MediaStream for frame extraction.
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("navigator.mediaDevices.getUserMedia is not available.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      startWebSocket();
      startFrameCaptureLoop();
      isCameraRunningRef.current = true;
      startVoiceRecognition();

      document.body.classList.add("camera-running");
      setIsCameraRunning(true);
    } catch (err) {
      document.body.classList.remove("camera-running");
      stopFrameCaptureLoop();
      stopMediaStream();
      stopWebSocket();
      stopAudioEngine();
      stopVoiceRecognition();
      isCameraRunningRef.current = false;
      setIsCameraRunning(false);
      setCameraError(
        err instanceof Error ? err.message : "Failed to start camera.",
      );
    }
  };

  const stopCamera = async () => {
    try {
      document.body.classList.remove("camera-running");
      stopFrameCaptureLoop();
      stopMediaStream();
      stopWebSocket();
      stopAudioEngine();
      stopVoiceRecognition();
      isCameraRunningRef.current = false;
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
      document.body.classList.remove("camera-running");
      stopFrameCaptureLoop();
      stopMediaStream();
      stopWebSocket();
      stopAudioEngine();
      stopVoiceRecognition();
      isCameraRunningRef.current = false;
      CameraView.stop().catch(() => {
        // noop - app might already be stopped/unmounted
      });
    };
  }, []);

  return (
    <main className="cv-page camera-modal">
      <header className="cv-header">
        <h1>Camera</h1>
        <p>{statusText}</p>
      </header>

      <section className="cv-controls" aria-label="Camera controls">
        <button onClick={startCamera} disabled={isCameraRunning}>
          Start Camera
        </button>
        <button onClick={stopCamera} disabled={!isCameraRunning}>
          Stop Camera
        </button>
        <label>
          FPS
          <input
            type="number"
            min={1}
            max={30}
            value={captureFps}
            onChange={(event) => {
              const next = Number(event.target.value);
              if (Number.isFinite(next)) {
                setCaptureFps(next);
              }
            }}
            disabled={isCameraRunning}
            style={{ width: 72, marginLeft: 8 }}
          />
        </label>
      </section>

      {cameraError ? <p className="cv-error">{cameraError}</p> : null}
      {backendError ? <p className="cv-error">{backendError}</p> : null}
      <p>{voiceStatus}</p>
      <p>
        Detections: {latestDetections.length} | Dist points: {getDistPointsToRender().length}
      </p>

      <section className="cv-layout">
        <div className="cv-camera-panel">
          <h2>Camera Preview</h2>
          <div id="cameraNativeContainer" style={{ display: "none" }} />
        <div id="cameraContainer" className="camera-container">
            <video
              ref={videoRef}
              style={{ width: "100%", height: "100%", objectFit: "cover" }}
              playsInline
              muted
              autoPlay
            />
            <canvas
              ref={overlayRef}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: "100%",
                pointerEvents: "none",
              }}
            />
          </div>
        </div>
      </section>
    </main>
  );
}
