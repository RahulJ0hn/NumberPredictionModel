import { forwardRef, useImperativeHandle, useRef, useState } from "react";

const CANVAS_SIZE = 280;

const DrawingCanvas = forwardRef(function DrawingCanvas(_props, ref) {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);

  const getContext = () => canvasRef.current.getContext("2d");

  const clear = () => {
    const ctx = getContext();
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  };

  useImperativeHandle(ref, () => ({
    clear,
    toDataURL: () => canvasRef.current.toDataURL("image/png"),
    isEmpty: () => {
      const ctx = getContext();
      const { data } = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
      for (let i = 0; i < data.length; i += 4) {
        if (data[i] !== 0 || data[i + 1] !== 0 || data[i + 2] !== 0) return false;
      }
      return true;
    },
  }));

  const pointerPos = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: clientX - rect.left, y: clientY - rect.top };
  };

  const startDrawing = (e) => {
    e.preventDefault();
    const ctx = getContext();
    const { x, y } = pointerPos(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
    setIsDrawing(true);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    e.preventDefault();
    const ctx = getContext();
    const { x, y } = pointerPos(e);
    ctx.lineTo(x, y);
    ctx.strokeStyle = "white";
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.stroke();
  };

  const stopDrawing = () => setIsDrawing(false);

  return (
    <canvas
      ref={(node) => {
        canvasRef.current = node;
        if (node && !node.dataset.initialized) {
          node.dataset.initialized = "true";
          const ctx = node.getContext("2d");
          ctx.fillStyle = "black";
          ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
        }
      }}
      width={CANVAS_SIZE}
      height={CANVAS_SIZE}
      className="drawing-canvas"
      onMouseDown={startDrawing}
      onMouseMove={draw}
      onMouseUp={stopDrawing}
      onMouseLeave={stopDrawing}
      onTouchStart={startDrawing}
      onTouchMove={draw}
      onTouchEnd={stopDrawing}
    />
  );
});

export default DrawingCanvas;
