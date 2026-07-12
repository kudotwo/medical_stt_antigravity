# 🐧 Linux Setup Guide — Antigravity Medical STT

This guide explains how to set up the project on a Linux machine after cloning from GitHub.

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/kudotwo/medical_stt_antigravity.git
cd medical_stt_antigravity
```

---

## Step 2 — Install System Dependencies

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv ffmpeg -y
```

> `ffmpeg` is required by Whisper to process audio files.

---

## Step 3 — Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Step 4 — Install Python Packages

```bash
cd "Source Code"
pip install -r requirements.txt
```

> **If you have an NVIDIA GPU**, install PyTorch with CUDA support instead:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Step 5 — Recreate Your `.env` File

The `.env` file is not included in the repository for security reasons. Create it manually:

```bash
nano "Source Code/.env"
```

Add the following lines with your actual API keys:

```
GEMINI_API_KEY = your_gemini_api_key_here
HF_TOKEN = your_huggingface_token_here
```

Save and exit: `Ctrl+O` → `Enter` → `Ctrl+X`

---

## Step 6 — Run the Pipeline

```bash
cd "Source Code"
python stt_pipeline.py
```

---

## ⚠️ Notes

- **Audio files** (`.wav`, `.mp3`, etc.) are excluded from GitHub due to their size.
  Copy them manually to Linux via USB drive or `scp`:
  ```bash
  scp -r /path/to/audio_sample user@linux-machine:~/medical_stt_antigravity/
  ```
- **Whisper models** will be downloaded automatically on first run (~1–3 GB depending on model size).
- Make sure your virtual environment is always activated before running scripts:
  ```bash
  source venv/bin/activate
  ```
