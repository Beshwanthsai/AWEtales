"""
Web/API Interface
Provides REST and WebSocket endpoints for the pipeline.
"""

from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
from unified_neural_pipeline import UnifiedNeuralPipeline

app = FastAPI(title="Unified Neural Pipeline API")

config = {
    "sample_rate": 16000,
    "output_dir": "output"
}
pipeline = UnifiedNeuralPipeline(config)

@app.post("/process")
async def process_audio(mixture: UploadFile = File(...), target: UploadFile = File(...)):
    # Save uploaded files temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as mix_file:
        mix_file.write(await mixture.read())
        mix_path = mix_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tgt_file:
        tgt_file.write(await target.read())
        tgt_path = tgt_file.name

    try:
        result = pipeline.process(mix_path, tgt_path)
        return JSONResponse(content=result)
    finally:
        os.unlink(mix_path)
        os.unlink(tgt_path)

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Placeholder for streaming implementation
    await websocket.send_text("Streaming not yet implemented")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)