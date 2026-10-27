import { useRef, useState } from "react";
import DrawingCanvas from "./DrawingCanvas";
import "./index.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  const canvasRef = useRef(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setError(null);

    if (canvasRef.current.isEmpty()) {
      setError("Draw a digit first.");
      return;
    }

    setLoading(true);
    try {
      const image = canvasRef.current.toDataURL();
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${response.status})`);
      }

      setResult(await response.json());
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    canvasRef.current.clear();
    setResult(null);
    setError(null);
  };

  const probabilities = result?.probabilities ?? Array(10).fill(0);

  return (
    <div className="app">
      <header>
        <h1>Digit Prediction</h1>
        <p className="subtitle">Draw a digit from 0 to 9, then predict.</p>
      </header>

      <div className="panel">
        <div className="draw">
          <DrawingCanvas ref={canvasRef} />
          <div className="controls">
            <button type="button" onClick={handlePredict} disabled={loading}>
              {loading ? "Predicting…" : "Predict"}
            </button>
            <button type="button" onClick={handleClear} className="secondary">
              Clear
            </button>
          </div>
          {error && <p className="error">{error}</p>}
        </div>

        <div className="readout" aria-live="polite">
          <p className="readout-label">Prediction</p>
          <div className="result">
            <span className={result ? "result-digit" : "result-digit placeholder"}>
              {result ? result.digit : "–"}
            </span>
            <span className="result-confidence">
              {result ? `${(result.confidence * 100).toFixed(1)}%` : "Draw to start"}
            </span>
          </div>
          <ul className="bars">
            {probabilities.map((p, digit) => (
              <li
                key={digit}
                className={result && digit === result.digit ? "bar-row winner" : "bar-row"}
              >
                <span>{digit}</span>
                <span className="bar-track">
                  <span className="bar-fill" style={{ width: `${p * 100}%` }} />
                </span>
                <span>{result ? `${(p * 100).toFixed(0)}%` : ""}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default App;
