# ── HuggingFace Spaces deployment ─────────────────────────────────────────────
# Uses Python 3.11 slim to keep the image small (no Whisper/torch needed).
# The app runs on port 7860 which is the HF Spaces default.

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy and install lightweight demo dependencies first (layer caching)
COPY "Source Code/requirements-demo.txt" .
RUN pip install --no-cache-dir -r requirements-demo.txt

# Copy the rest of the Source Code folder
COPY "Source Code/" .

# HuggingFace Spaces requires the app to run on port 7860
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
