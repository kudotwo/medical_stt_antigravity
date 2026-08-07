import sys
import os

# Use abspath so dirname() never returns an empty string when __file__ is relative
_HERE = os.path.dirname(os.path.abspath(__file__))

# Add "Source Code/" to the Python path so server.py and its imports are found
sys.path.insert(0, os.path.join(_HERE, "..", "Source Code"))

# Import the FastAPI app — Vercel looks for a variable named `app` in this file
# Note: no os.chdir() needed — server.py uses __file__-relative paths for everything
from server import app  # noqa: F401, E402
