# =====================================================================================
# UI image: Streamlit dashboard (small, no TensorFlow)
#
# Build:  docker build -t corn-disease-ui ./ui
# Run:    docker run -p 8501:8501 -e API_URL=http://host.docker.internal:8000 corn-disease-ui
# =====================================================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8501 \
    API_URL=http://localhost:8000

WORKDIR /app
RUN useradd --create-home --uid 1000 appuser

COPY requirements.txt .
RUN pip install -r requirements.txt

# The app, its theme (.streamlit/config.toml: colours) and its logo
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser .streamlit/ .streamlit/
COPY --chown=appuser:appuser assets/ assets/
USER appuser
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.getenv(\"PORT\", \"8501\")}/_stcore/health', timeout=4)"

# headless = don't try to open a browser; address 0.0.0.0 = reachable from outside the container
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false"]
