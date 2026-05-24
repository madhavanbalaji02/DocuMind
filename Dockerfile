FROM python:3.11-slim

# Add PostgreSQL official repo so postgresql-15 is guaranteed available
RUN apt-get update && apt-get install -y curl ca-certificates gnupg lsb-release && \
    curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc \
        | gpg --dearmor -o /usr/share/keyrings/pgdg-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/pgdg-keyring.gpg] \
        https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
        > /etc/apt/sources.list.d/pgdg.list && \
    apt-get update && apt-get install -y \
        postgresql-15 \
        postgresql-client-15 \
        supervisor \
        wget \
        gosu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY src/ ./src/

RUN pip install --no-cache-dir --prefer-binary .

# Pre-download embedding model into a fixed cache dir so the API starts instantly
ENV HF_HOME=/app/.hf_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY supervisord.conf /etc/supervisor/supervisord.conf
COPY startup.sh /startup.sh
RUN chmod +x /startup.sh && \
    mkdir -p /data/qdrant /data/postgres /var/log/supervisor /app/logs \
             /var/run/postgresql && \
    chown postgres:postgres /var/run/postgresql

ENV DATABASE_URL_ASYNC=postgresql://documind:documind_secret@localhost:5432/documind \
    QDRANT_LOCAL_PATH=/data/qdrant \
    QDRANT_COLLECTION=documind_chunks \
    EMBEDDING_MODE=local \
    API_HOST=0.0.0.0 \
    API_PORT=8888

EXPOSE 7860

CMD ["/startup.sh"]
