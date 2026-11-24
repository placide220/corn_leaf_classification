# =====================================================================================
# API image: FastAPI + TensorFlow model
#
# Build:  docker build -t corn-disease-api .
# Run:    docker run -p 8000:8000 corn-disease-api
#
# The model (models/) and the train/test images (data/) are copied INTO the image, so a
# single container works on its own (e.g. on a cloud platform). With docker-compose these
# folders are replaced by shared volumes so all replicas see the same model and data.
# =====================================================================================
FROM python:3.12-slim

# - PYTHONUNBUFFERED: print logs straight away (no buffering) so `docker logs` is live
# - TF_CPP_MIN_LOG_LEVEL=2: hide TensorFlow's info/warning spam in the logs
# - PORT: the port uvicorn listens on. Cloud platforms often set this themselves.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8000

WORKDIR /app

# Run as a normal user (uid 1000), not root. Safer, and required by some hosts such as
# Hugging Face Spaces.
RUN useradd --create-home --uid 1000 appuser

# Install dependencies FIRST: Docker caches this layer, so changing your code later
# does not re-download TensorFlow on every build.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Then copy the application code, the trained model and the data used for retraining
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser api/ api/
COPY --chown=appuser:appuser models/ models/
COPY --chown=appuser:appuser data/ data/
RUN mkdir -p data/uploads models/archive && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

# Docker marks the container "healthy" only once /health answers. docker-compose waits for
# this before starting nginx. start-period gives TensorFlow time to load the model.
HEALTHCHECK --interval=15s --timeout=60s --start-period=90s --retries=5 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.getenv(\"PORT\", \"8000\")}/health', timeout=55)"

# One uvicorn worker per container: in the load test we scale by adding CONTAINERS, so each
# replica is one comparable unit. (sh -c is needed so ${PORT} is expanded.)
# --timeout-keep-alive 75: keep idle connections open LONGER than nginx does (60 s, see
# nginx.conf). Otherwise uvicorn can close a connection at the moment nginx reuses it,
# and that request fails with a 502 under heavy load.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 75"]
