from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import io

# Import the function from our existing pipeline
from stt_pipeline import diarize_and_extract_soap_from_text, _flatten

app = FastAPI(title="Medical STT Live API")

# Serve the static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

class AnalyzeRequest(BaseModel):
    text: str

class CSVRequest(BaseModel):
    soap_data: dict

@app.get("/")
async def root():
    # Redirect root to our index.html
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/analyze")
async def analyze_text(request: AnalyzeRequest):
    """
    Takes the raw transcript text from the frontend, sends it to Gemini for 
    Diarization and SOAP extraction, and returns the structured JSON.
    """
    if not request.text.strip():
        return JSONResponse(status_code=400, content={"error": "Empty text provided"})
        
    result = diarize_and_extract_soap_from_text(request.text)
    
    if result is None:
        return JSONResponse(status_code=500, content={"error": "Failed to extract SOAP report."})
        
    return result

@app.post("/api/download_csv")
async def download_csv(request: CSVRequest):
    """
    Takes the structured SOAP JSON, flattens it using our existing pipeline logic,
    and returns a raw CSV string for the frontend to download.
    """
    soap_dict = request.soap_data
    # Remove diarized_segments for the flat CSV report
    soap_clean = {k: v for k, v in soap_dict.items() if k != 'diarized_segments'}
    
    # Flatten using pandas (same logic as in save_results)
    flat = pd.json_normalize(soap_clean, sep='.')
    flat = _flatten(flat)
    
    # We don't have an audio file name for live STT, so use 'live_recording'
    flat.insert(0, 'audio_file', 'live_recording')
    
    # Convert to CSV string
    output = io.StringIO()
    flat.to_csv(output, index=False)
    
    return PlainTextResponse(content=output.getvalue(), media_type="text/csv")

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    print("Starting Medical STT Live Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
