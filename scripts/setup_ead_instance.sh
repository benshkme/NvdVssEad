#!/usr/bin/env bash
# =============================================================================
# EAD Instance Setup Script — g6e.12xlarge (L40S, 2-GPU local deployment)
# =============================================================================
#
# Run once on a fresh EC2 instance before the first deployment.
# Safe to re-run: it updates existing values rather than duplicating them.
#
# USAGE
#   cd ~/NvdVssEad
#   bash scripts/setup_ead_instance.sh
#
# WHAT IT DOES
#   1. Detects private + public IPs and patches them into dev-profile-ead/.env
#   2. Prompts for your NGC API key and stores it in ~/.ngc_api_key (chmod 600)
#   3. Adds shell functions to ~/.bashrc:
#        run_vss_ead       — full stack deployment (tears down & redeploys)
#        start_vst         — start only the VST containers (no teardown)
#        rebuild_ead_agent — rebuild the custom vss-agent-ead Docker image
#        ead_status        — show status of all EAD-profile containers
#
# SECURITY GROUPS (open in AWS Console before deploying)
#   Port 22    — SSH
#   Port 3000  — UI
#   Port 8000  — Agent API / WebSocket
#   Port 30888 — VST video upload
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_DIR/deployments/developer-workflow/dev-profile-ead/.env"
NGC_KEY_FILE="$HOME/.ngc_api_key"
BASHRC="$HOME/.bashrc"

# Colour helpers
_info()    { echo -e "\e[32m[INFO]\e[0m  $*"; }
_warn()    { echo -e "\e[33m[WARN]\e[0m  $*"; }
_section() { echo -e "\n\e[1;34m=== $* ===\e[0m"; }

# =============================================================================
# 1. Detect IPs
# =============================================================================
_section "Detecting network addresses"

PRIVATE_IP=$(hostname -I | awk '{print $1}')
PUBLIC_IP=$(curl -s --max-time 8 ifconfig.me 2>/dev/null || \
            curl -s --max-time 8 icanhazip.com 2>/dev/null || \
            echo "")

if [[ -z "$PUBLIC_IP" ]]; then
  _warn "Could not auto-detect public IP. Enter it manually:"
  read -r -p "  Public IP: " PUBLIC_IP
fi

_info "Private IP : $PRIVATE_IP"
_info "Public IP  : $PUBLIC_IP"

# =============================================================================
# 2. Patch .env
# =============================================================================
_section "Updating dev-profile-ead/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: .env not found at $ENV_FILE — is the repo cloned correctly?" >&2
  exit 1
fi

_patch_env() {
  local key="$1" val="$2"
  if grep -q "^${key}=" "$ENV_FILE"; then
    sed -i "s|^${key}=.*|${key}=${val}|" "$ENV_FILE"
    _info "Set ${key}=${val}"
  else
    echo "${key}=${val}" >> "$ENV_FILE"
    _info "Added ${key}=${val}"
  fi
}

_patch_env "HOST_IP"     "'${PRIVATE_IP}'"
_patch_env "EXTERNAL_IP" "${PUBLIC_IP}"

# =============================================================================
# 3. NGC API key
# =============================================================================
_section "NGC API key"

# Read the key directly from the .env file (NGC_CLI_API_KEY is set inline there).
# The setup script no longer manages a separate ~/.ngc_api_key file.
ENV_KEY=$(grep "^NGC_CLI_API_KEY=" "$ENV_FILE" | cut -d"'" -f2 | tr -d '[:space:]')

if [[ "$ENV_KEY" =~ ^nvapi-[A-Za-z0-9_-]{40,}$ ]]; then
  _info "NGC_CLI_API_KEY found in .env — no prompt needed."
  # Keep ~/.ngc_api_key in sync for run_vss_ead compatibility
  printf '%s' "$ENV_KEY" > "$NGC_KEY_FILE"
  chmod 600 "$NGC_KEY_FILE"
else
  _warn "NGC_CLI_API_KEY in .env is missing or invalid (value: '${ENV_KEY:0:12}...')."
  _warn "Edit $ENV_FILE and set NGC_CLI_API_KEY='nvapi-...' then re-run this script."
fi

# =============================================================================
# 4. Shell functions in ~/.bashrc
# =============================================================================
_section "Adding shell functions to ~/.bashrc"

FUNCTIONS_BLOCK=$(cat <<'FUNCBLOCK'
# ----------------------------------------------------------------------------
# EAD deployment helpers (added by scripts/setup_ead_instance.sh)
# ----------------------------------------------------------------------------

# Full stack deployment (tears down existing stack, rebuilds EAD agent image,
# and brings everything up fresh — expect ~10 min on first run for NIM warmup)
run_vss_ead() {
  local repo="$HOME/NvdVssEad"
  local key_file="$HOME/.ngc_api_key"
  if [[ ! -f "$key_file" || ! -s "$key_file" ]]; then
    echo "[ERROR] NGC API key not found at $key_file"
    echo "        Run: bash $repo/scripts/setup_ead_instance.sh"
    return 1
  fi
  NGC_CLI_API_KEY="$(cat "$key_file")" bash "$repo/scripts/dev-profile.sh" up \
    --profile ead \
    --hardware-profile L40S \
    --host-ip "$(hostname -I | awk '{print $1}')" \
    --llm-device-id 0 \
    --vlm-device-id 1
}

# Start only the VST containers without tearing down the running stack.
# Use this when streamprocessing-ms-dev failed its initial health check
# but is now running fine and you just need the dependent VST services to start.
start_vst() {
  local repo="$HOME/NvdVssEad"
  local env_file="$repo/deployments/developer-workflow/dev-profile-ead/generated.env"
  if [[ ! -f "$env_file" ]]; then
    echo "[ERROR] generated.env not found — run run_vss_ead first"
    return 1
  fi
  cd "$repo/deployments" && docker compose \
    --env-file "developer-workflow/dev-profile-ead/generated.env" \
    start \
    streamprocessing-ms-dev \
    sdr-streamprocessing \
    envoy-streamprocessing \
    sensor-ms-dev \
    vst-ingress-dev \
    vst-mcp-dev
}

# Rebuild the custom vss-agent-ead Docker image after updating EAD tool source.
# The image is cached so this only needs to run when agent/src/ changes.
rebuild_ead_agent() {
  local repo="$HOME/NvdVssEad"
  local version
  version=$(grep "^VSS_AGENT_VERSION=" \
    "$repo/deployments/developer-workflow/dev-profile-ead/.env" 2>/dev/null \
    | cut -d'=' -f2- | tr -d '"' | head -1)
  version="${version:-3.1.0}"
  echo "[INFO] Removing cached image vss-agent-ead:${version} ..."
  docker rmi "vss-agent-ead:${version}" 2>/dev/null || true
  echo "[INFO] Building vss-agent-ead:${version} ..."
  docker build \
    -f "$repo/deployments/developer-workflow/dev-profile-ead/Dockerfile.vss-agent" \
    -t "vss-agent-ead:${version}" \
    --build-arg "VSS_AGENT_VERSION=${version}" \
    "$repo"
  echo "[INFO] Done: vss-agent-ead:${version}"
}

# Show the status of all EAD-profile containers at a glance.
ead_status() {
  docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" \
    | grep -E "NAMES|vss-agent|metropolis|streamprocessing|sensor-ms|sdr-stream|envoy-stream|vst-|lvs-server|mdx-|cosmos|nemotron|phoenix"
}
FUNCBLOCK
)

MARKER="# EAD deployment helpers (added by scripts/setup_ead_instance.sh)"

if grep -qF "$MARKER" "$BASHRC" 2>/dev/null; then
  _info "Shell functions already present in $BASHRC — skipping."
else
  printf '\n%s\n' "$FUNCTIONS_BLOCK" >> "$BASHRC"
  _info "Shell functions added to $BASHRC"
fi

# =============================================================================
# Done
# =============================================================================
_section "Setup complete"

cat <<EOF

Available commands (after reloading your shell):

  run_vss_ead        Deploy the full EAD stack (tears down & redeploys)
  start_vst          Start VST containers only — no teardown of running stack
  rebuild_ead_agent  Rebuild the custom vss-agent-ead image after code changes
  ead_status         Show status of all EAD containers

Reload your shell now:
  source ~/.bashrc

Then deploy:
  run_vss_ead

Remember to open these ports in your EC2 Security Group:
  22    SSH
  3000  UI
  8000  Agent API + WebSocket
  30888 VST video upload

EOF
