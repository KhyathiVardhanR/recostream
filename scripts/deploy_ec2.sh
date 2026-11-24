#!/usr/bin/env bash
# Ship the code to the EC2 instance created by deploy/terraform and start the stack.
#   scripts/deploy_ec2.sh <public-ip> [ssh-key.pem]
# Train locally (or on the instance) with S3_BUCKET set so the model lands in S3:
#   S3_BUCKET=$(terraform -chdir=deploy/terraform output -raw bucket) python -m recostream.train
set -euo pipefail
HOST=$1
KEY=${2:-~/.ssh/id_rsa}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
rsync -az --exclude .venv --exclude data --exclude artifacts --exclude .git -e "ssh -i $KEY" \
  "$ROOT/" "ec2-user@$HOST:~/recostream/"
ssh -i "$KEY" "ec2-user@$HOST" 'cd ~/recostream && set -a && . /etc/environment && set +a && \
  docker compose up -d --build api consumer kafka redis prometheus && docker compose ps'
echo "API: http://$HOST:8000/healthz   Prometheus: http://$HOST:9090"
