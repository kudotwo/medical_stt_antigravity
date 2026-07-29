import json, os, io
from datetime import datetime, timedelta, timezone
from pathlib import Path

import bcrypt
import jwt
import pandas as pd
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Import the function from our existing pipeline
from stt_pipeline import diarize_and_extract_soap_from_text, _flatten

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
load_dotenv()
JWT_SECRET   = os.environ.get("JWT_SECRET_KEY", "change-me-in-env")
JWT_ALGO     = "HS256"
JWT_EXPIRY   = timedelta(hours=12)       # re-login every 12 hours
USERS_FILE   = Path(__file__).parent / "users.json"
COOKIE_NAME  = "stt_session"

# ── Login toggle ──────────────────────────────────────────────────────────────
# Set ENABLE_LOGIN=false in .env to bypass the login page entirely.
# Set ENABLE_LOGIN=true  to enforce login with demo/premium accounts.
ENABLE_LOGIN = os.environ.get("ENABLE_LOGIN", "true").strip().lower() == "true"
print(f"[Config] Login required : {ENABLE_LOGIN}")

app = FastAPI(title="Medical STT Live API")

# Serve the static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ─────────────────────────────────────────────
#  User store helpers
# ─────────────────────────────────────────────

# Default users — pre-hashed with bcrypt.
# These are the credentials used if users.json does not exist yet
# (e.g. on a fresh Render deployment where the file isn't committed to git).
_DEFAULT_USERS = {
    "demo_user": {
        "password_hash": "$2b$12$vgZZk6.95gQ1EFre3e65b.7gr7I3ymTqNWoiSMh.tn3fwkVLfg0ki",
        "role": "demo",
        "first_login": None,
        "expires_days": 7,
    },
    "premium_user": {
        "password_hash": "$2b$12$JTgyms30l69cPe38qwT/XeehkOp9rzGg6BGeWx.I3Qm5k/ht2OlNu",
        "role": "premium",
        "first_login": None,
        "expires_days": None,
    },
}

def _ensure_users_file() -> None:
    """Create users.json with default credentials if it doesn't exist.
    Called once at startup — handles fresh Render deployments automatically.
    """
    if not USERS_FILE.exists():
        print(f"[Auth] users.json not found — creating with default accounts at {USERS_FILE}")
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(_DEFAULT_USERS, f, indent=2)
        print("[Auth] users.json created ✓")
    else:
        print(f"[Auth] users.json found at {USERS_FILE}")

_ensure_users_file()  # Run at import time


def _load_users() -> dict:
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_users(users: dict) -> None:
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


# ─────────────────────────────────────────────
#  Auth helpers
# ─────────────────────────────────────────────
def _issue_token(username: str, role: str) -> str:
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + JWT_EXPIRY,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def _verify_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except jwt.PyJWTError:
        return None


def get_current_user(request: Request) -> dict:
    """FastAPI dependency — returns the decoded JWT payload or raises 401.
    When ENABLE_LOGIN is False, always passes through as a guest user.
    """
    if not ENABLE_LOGIN:
        return {"sub": "guest", "role": "premium"}
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = _verify_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Session expired")
    return payload


# ─────────────────────────────────────────────
#  Pydantic models
# ─────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str


class AnalyzeRequest(BaseModel):
    text: str


class CSVRequest(BaseModel):
    soap_data: dict


# ─────────────────────────────────────────────
#  Auth routes
# ─────────────────────────────────────────────
@app.get("/login")
async def login_page():
    with open("static/login.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/login")
async def api_login(body: LoginRequest):
    users = _load_users()
    user_data = users.get(body.username)

    # --- Validate credentials ---
    if user_data is None:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    pw_bytes = body.password.encode("utf-8")
    stored_hash = user_data["password_hash"].encode("utf-8")
    if not bcrypt.checkpw(pw_bytes, stored_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    # --- Check demo account expiry ---
    role = user_data["role"]
    if role == "demo":
        now = datetime.now(timezone.utc)
        if user_data["first_login"] is None:
            # First login — stamp the timestamp
            user_data["first_login"] = now.isoformat()
            users[body.username] = user_data
            _save_users(users)
        else:
            first_login_dt = datetime.fromisoformat(user_data["first_login"])
            expires_days   = user_data.get("expires_days", 7)
            expiry_dt      = first_login_dt + timedelta(days=expires_days)
            if now >= expiry_dt:
                raise HTTPException(
                    status_code=403,
                    detail="Your demo access has expired. Please contact us to upgrade."
                )

    # --- Issue token ---
    token = _issue_token(body.username, role)
    response = JSONResponse({"message": "Login successful", "role": role})
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=False,       # set True in production behind HTTPS (Render handles this)
        samesite="lax",
        max_age=int(JWT_EXPIRY.total_seconds()),
    )
    return response


@app.post("/api/logout")
async def api_logout():
    response = JSONResponse({"message": "Logged out"})
    response.delete_cookie(COOKIE_NAME)
    return response


# ─────────────────────────────────────────────
#  Protected routes
# ─────────────────────────────────────────────
@app.get("/")
async def root(request: Request):
    """Redirect root → login if not authenticated, else → /app.
    When ENABLE_LOGIN is False, serves the app directly.
    """
    if not ENABLE_LOGIN:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    token = request.cookies.get(COOKIE_NAME)
    if token and _verify_token(token):
        return RedirectResponse(url="/app")
    return RedirectResponse(url="/login")


@app.get("/app")
async def main_app(_user: dict = Depends(get_current_user)):
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/analyze")
async def analyze_text(request: AnalyzeRequest, _user: dict = Depends(get_current_user)):
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
async def download_csv(request: CSVRequest, _user: dict = Depends(get_current_user)):
    """
    Takes the structured SOAP JSON, flattens it using our existing pipeline logic,
    and returns a raw CSV string for the frontend to download.
    """
    soap_dict = request.soap_data
    # Remove diarized_segments for the flat CSV report
    soap_clean = {k: v for k, v in soap_dict.items() if k != "diarized_segments"}

    # Flatten using pandas (same logic as in save_results)
    flat = pd.json_normalize(soap_clean, sep=".")
    flat = _flatten(flat)

    # We don't have an audio file name for live STT, so use 'live_recording'
    flat.insert(0, "audio_file", "live_recording")

    # Convert to CSV string
    output = io.StringIO()
    flat.to_csv(output, index=False)

    return PlainTextResponse(content=output.getvalue(), media_type="text/csv")


# ─────────────────────────────────────────────
#  API: return current user info (for the frontend)
# ─────────────────────────────────────────────
@app.get("/api/me")
async def me(user: dict = Depends(get_current_user)):
    return {"username": user["sub"], "role": user["role"]}


if __name__ == "__main__":
    import uvicorn
    print("Starting Medical STT Live Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
