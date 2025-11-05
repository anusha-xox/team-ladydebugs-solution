
FROM python:3.11-slim


ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1


WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt


COPY . .


RUN chmod +x start.sh


EXPOSE 8501


ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

ENV PORT=8501


ENTRYPOINT ["./start.sh"]
