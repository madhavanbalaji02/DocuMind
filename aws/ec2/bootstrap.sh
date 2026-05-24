#!/bin/bash
# DocuMind EC2 bootstrap — runs as user-data on first launch.
# Installs: Python 3.11, uvicorn, nginx, systemd services.
set -e

# ── System setup ──────────────────────────────────────────────────────────────
yum update -y
yum install -y python3.11 python3.11-pip git nginx

# ── Clone repo ────────────────────────────────────────────────────────────────
git clone https://github.com/madhavanbalaji02/DocuMind /opt/documind || \
  (cd /opt/documind && git pull)
cd /opt/documind

# ── Install Python deps ───────────────────────────────────────────────────────
pip3.11 install uv
uv pip install -e . --system --python python3.11

# ── Load secrets from SSM ─────────────────────────────────────────────────────
python3.11 - <<'PYEOF'
import boto3, os
ssm = boto3.client('ssm', region_name='us-east-2')
paginator = ssm.get_paginator('get_parameters_by_path')
lines = []
for page in paginator.paginate(Path='/documind', WithDecryption=True):
    for p in page['Parameters']:
        key = p['Name'].split('/')[-1].upper().replace('-', '_')
        lines.append(f'{key}={p["Value"]}')
with open('/opt/documind/.env', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'Loaded {len(lines)} parameters from SSM')
PYEOF

# ── systemd: FastAPI backend ──────────────────────────────────────────────────
cat > /etc/systemd/system/documind.service << 'EOF'
[Unit]
Description=DocuMind FastAPI Backend
After=network.target

[Service]
WorkingDirectory=/opt/documind
EnvironmentFile=/opt/documind/.env
ExecStart=/usr/local/bin/uvicorn src.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 2 \
  --log-level info
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# ── systemd: Streamlit UI ─────────────────────────────────────────────────────
cat > /etc/systemd/system/documind-ui.service << 'EOF'
[Unit]
Description=DocuMind Streamlit UI
After=network.target documind.service

[Service]
WorkingDirectory=/opt/documind
EnvironmentFile=/opt/documind/.env
ExecStart=/usr/local/bin/streamlit run src/ui/app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.baseUrlPath=/ui \
  --server.headless=true
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# ── nginx reverse proxy ───────────────────────────────────────────────────────
cat > /etc/nginx/conf.d/documind.conf << 'EOF'
server {
    listen 80;
    server_name _;

    # FastAPI
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
        proxy_connect_timeout 30s;
    }

    # Streamlit UI
    location /ui {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }
}
EOF

# Disable default nginx config
rm -f /etc/nginx/conf.d/default.conf

# ── Start services ────────────────────────────────────────────────────────────
systemctl daemon-reload
systemctl enable --now documind
systemctl enable --now documind-ui
systemctl enable --now nginx

echo "DocuMind bootstrap complete!"
echo "API:  http://$(curl -s checkip.amazonaws.com)/health"
echo "UI:   http://$(curl -s checkip.amazonaws.com)/ui"
