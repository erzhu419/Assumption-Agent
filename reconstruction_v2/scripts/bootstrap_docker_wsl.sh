#!/usr/bin/env bash
set -euo pipefail

if [[ ! -r /proc/version ]] || ! grep -qi microsoft /proc/version; then
  echo "This bootstrap is intended for WSL2." >&2
  exit 2
fi

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemd/systemctl is required in this WSL distribution." >&2
  exit 2
fi

sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker "${USER}"
sudo docker info >/dev/null

echo "Docker is running. Start a new WSL shell so docker-group membership takes effect."
echo "Then run: docker info"
