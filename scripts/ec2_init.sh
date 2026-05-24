#!/usr/bin/env bash
# Remote init script — runs ON the EC2 instance.
# Idempotent: safe to re-run on every deploy.

set -euo pipefail
APP_DIR="/home/ubuntu/documind"
VENV="$APP_DIR/.venv"

echo "=== DocuMind EC2 init ==="

cd "$APP_DIR"

# ── Python venv + deps ────────────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
    echo "[deps] Creating venv…"
    python3.11 -m venv "$VENV"
fi
echo "[deps] Installing Python packages…"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -e "."

# ── Docker Compose (Qdrant only — production uses RDS for Postgres) ───────────
echo "[infra] Starting Qdrant via docker compose…"
# Rewrite docker-compose to only run Qdrant on EC2
cat > /tmp/docker-compose-ec2.yml <<'YAML'
services:
  qdrant:
    image: qdrant/qdrant:latest
    restart: unless-stopped
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - "/home/ubuntu/documind/data/qdrant:/qdrant/storage"
    environment:
      QDRANT__SERVICE__GRPC_PORT: "6334"
YAML
mkdir -p "$APP_DIR/data/qdrant"
docker compose -f /tmp/docker-compose-ec2.yml up -d

# ── nginx ─────────────────────────────────────────────────────────────────────
echo "[nginx] Configuring reverse proxy…"
sudo tee /etc/nginx/sites-available/documind > /dev/null <<'NGINX'
server {
    listen 80 default_server;
    server_name _;

    # Streamlit dashboard
    location / {
        proxy_pass         http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }

    # FastAPI — accessible at /api/ as well
    location /api/ {
        rewrite ^/api/(.*) /$1 break;
        proxy_pass         http://127.0.0.1:8888;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
}
NGINX

sudo ln -sf /etc/nginx/sites-available/documind /etc/nginx/sites-enabled/documind
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl restart nginx

# ── systemd: FastAPI ──────────────────────────────────────────────────────────
echo "[systemd] Writing documind-api.service…"
ENV_FILE="$APP_DIR/.env"
sudo tee /etc/systemd/system/documind-api.service > /dev/null <<SERVICE
[Unit]
Description=DocuMind FastAPI
After=network.target docker.service
Wants=docker.service

[Service]
User=ubuntu
WorkingDirectory=$APP_DIR
EnvironmentFile=-$ENV_FILE
Environment=PYTHONUNBUFFERED=1
ExecStart=$VENV/bin/uvicorn src.api.main:app --host 127.0.0.1 --port 8888 --workers 1
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICE

# ── systemd: Streamlit ────────────────────────────────────────────────────────
echo "[systemd] Writing documind-ui.service…"
sudo tee /etc/systemd/system/documind-ui.service > /dev/null <<SERVICE
[Unit]
Description=DocuMind Streamlit UI
After=documind-api.service

[Service]
User=ubuntu
WorkingDirectory=$APP_DIR
EnvironmentFile=-$ENV_FILE
Environment=PYTHONUNBUFFERED=1
Environment=API_BASE_URL=http://127.0.0.1:8888
ExecStart=$VENV/bin/streamlit run src/ui/app.py \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICE

# ── Enable + start ─────────────────────────────────────────────────────────────
echo "[systemd] Enabling services…"
sudo systemctl daemon-reload
sudo systemctl enable documind-api.service documind-ui.service
sudo systemctl restart documind-api.service documind-ui.service

echo ""
echo "=== Init complete ==="
echo "  Dashboard → http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)/"
echo "  API docs  → http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8888/docs"
echo ""
