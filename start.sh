set -e

export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_RUN_ON_SAVE=false
export STREAMLIT_SERVER_ADDRESS=0.0.0.0

PORT=${PORT:-8501}

exec streamlit run app.py --server.port "$PORT" --server.address 0.0.0.0