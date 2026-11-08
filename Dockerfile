FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MYDAILYNEWS_ALLOW_MODEL_DOWNLOAD=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .

RUN chmod +x docker-entrypoint.sh \
    && cp docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

EXPOSE 8765

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

CMD ["python", "gui.py", "--host", "0.0.0.0", "--port", "8765"]
