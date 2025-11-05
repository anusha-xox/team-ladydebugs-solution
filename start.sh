#!/usr/bin/env bash
set -e

# Ensure Streamlit runs headless and listens on all interfaces
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_RUN_ON_SAVE=false
export STREAMLIT_SERVER_ADDRESS=0.0.0.0

# If your platform (Code Engine) provides PORT, use it. Otherwise default to 8501.
PORT=${PORT:-8501}

exec streamlit run app.py --server.port "$PORT" --server.address 0.0.0.0
