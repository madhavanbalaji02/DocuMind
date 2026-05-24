#!/usr/bin/env bash
# Deploy DocuMind to EC2.
# Usage:
#   export EC2_IP=<public-ip>
#   bash scripts/deploy_to_ec2.sh
#
# Optional: export EC2_IP before calling, or pass it as $1.

set -euo pipefail

EC2_IP="${EC2_IP:-${1:-}}"
if [[ -z "$EC2_IP" ]]; then
    # Try reading from provisioner state file
    if [[ -f scripts/.ec2_state.json ]]; then
        EC2_IP=$(python3 -c "import json; print(json.load(open('scripts/.ec2_state.json'))['public_ip'])")
    fi
fi

if [[ -z "$EC2_IP" ]]; then
    echo "ERROR: Set EC2_IP or run provision_ec2.py first."
    exit 1
fi

KEY="$HOME/.ssh/documind-key.pem"
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15"
REMOTE="ubuntu@$EC2_IP"
APP_DIR="/home/ubuntu/documind"

echo ""
echo "=== Deploying DocuMind to $EC2_IP ==="
echo ""

# ── 1. Wait for SSH to become available ───────────────────────────────────────
echo "[1/5] Waiting for SSH…"
for i in $(seq 1 20); do
    if ssh $SSH_OPTS "$REMOTE" "echo ok" &>/dev/null; then
        echo "  SSH ready."
        break
    fi
    echo "  Attempt $i/20…"
    sleep 10
done

# ── 2. Sync project files ─────────────────────────────────────────────────────
echo "[2/5] Syncing project files…"
rsync -az --progress \
    -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
    --exclude='.venv' \
    --exclude='data/' \
    --exclude='logs/' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='hf_README.md' \
    . "$REMOTE:$APP_DIR/"

# ── 3. Copy .env if present locally ───────────────────────────────────────────
if [[ -f .env ]]; then
    echo "[3/5] Copying .env…"
    scp $SSH_OPTS .env "$REMOTE:$APP_DIR/.env"
else
    echo "[3/5] No .env found locally — you'll need to create one on EC2."
    echo "      ssh -i $KEY $REMOTE"
    echo "      nano $APP_DIR/.env"
fi

# ── 4. Run remote init (idempotent) ───────────────────────────────────────────
echo "[4/5] Running remote init script…"
ssh $SSH_OPTS "$REMOTE" "bash $APP_DIR/scripts/ec2_init.sh"

# ── 5. Restart services ───────────────────────────────────────────────────────
echo "[5/5] Restarting services…"
ssh $SSH_OPTS "$REMOTE" "
    sudo systemctl restart documind-api.service
    sudo systemctl restart documind-ui.service
    sudo systemctl status documind-api.service --no-pager -l
    sudo systemctl status documind-ui.service  --no-pager -l
"

echo ""
echo "=== Deployment complete ==="
echo ""
echo "  Dashboard  →  http://$EC2_IP/"
echo "  API docs   →  http://$EC2_IP:8888/docs"
echo ""
