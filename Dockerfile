FROM python:3.11-slim

# HuggingFace Spaces requires a non-root user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy and install lightweight demo dependencies first (Docker layer caching)
# JSON array syntax is required for paths that contain spaces
COPY --chown=user ["Source Code/requirements-demo.txt", "."]
RUN pip install --no-cache-dir -r requirements-demo.txt

# Copy all Source Code contents into the working directory
COPY --chown=user ["Source Code/", "."]

# HuggingFace Spaces default port
EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
