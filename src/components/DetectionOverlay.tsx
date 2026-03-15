import { memo } from "react";
import type { BBox } from "@/contexts/OnnxContext";

interface DetectionOverlayProps {
  boxes: BBox[];
  imageWidth: number;
  imageHeight: number;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

export const DetectionOverlay = memo(function DetectionOverlay({
  boxes,
  imageWidth,
  imageHeight,
}: DetectionOverlayProps) {
  if (!imageWidth || !imageHeight || boxes.length === 0) {
    return null;
  }

  return (
    <div className="detection-overlay" aria-hidden="true">
      {boxes.map((box, index) => {
        const left = clamp((box.xMin / imageWidth) * 100, 0, 100);
        const top = clamp((box.yMin / imageHeight) * 100, 0, 100);
        const width = clamp(((box.xMax - box.xMin) / imageWidth) * 100, 0, 100);
        const height = clamp(
          ((box.yMax - box.yMin) / imageHeight) * 100,
          0,
          100,
        );

        return (
          <div
            key={`${box.classIndex}-${box.score}-${index}`}
            className="detection-box"
            style={{
              left: `${left}%`,
              top: `${top}%`,
              width: `${width}%`,
              height: `${height}%`,
            }}
          >
            <span className="detection-label">
              {`cls ${box.classIndex} • ${(box.score * 100).toFixed(1)}%`}
            </span>
          </div>
        );
      })}
    </div>
  );
});
