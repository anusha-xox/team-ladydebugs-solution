# ===========================
#  Stage 1 — Base Image Setup
# ===========================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies (only minimal ones)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file and install dependencies first (for caching)
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of your app code
COPY . .

# Make sure start.sh is executable
RUN chmod +x start.sh

# Expose the default Streamlit port
EXPOSE 8501

# Default environment variables for Streamlit (override via Code Engine if needed)
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Default port (Code Engine will override this)
ENV PORT=8501

# Run Streamlit app via start.sh
ENTRYPOINT ["./start.sh"]
