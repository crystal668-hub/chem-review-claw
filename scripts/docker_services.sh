#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINERU_DIR="$ROOT_DIR/mineru-api-docker"
GROBID_DIR="$ROOT_DIR/grobid-docker"

usage() {
  cat <<'EOF'
Usage:
  scripts/docker_services.sh up [--build]
  scripts/docker_services.sh down
  scripts/docker_services.sh restart [--build]
  scripts/docker_services.sh ps
  scripts/docker_services.sh logs [grobid|mineru-api]
  scripts/docker_services.sh health
  scripts/docker_services.sh config

Commands:
  up       Start all project-required Docker services.
  down     Stop all project-required Docker services.
  restart  Restart all project-required Docker services.
  ps       Show container status for all project-required Docker services.
  logs     Tail logs for one service or all services.
  health   Run lightweight HTTP health checks.
  config   Render and validate Docker Compose configuration.
EOF
}

run_compose() {
  local dir="$1"
  shift
  docker compose --project-directory "$dir" -f "$dir/compose.yaml" "$@"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

wait_for_http() {
  local name="$1"
  local url="$2"
  local attempts="${3:-30}"
  local sleep_seconds="${4:-2}"
  local i

  for ((i = 1; i <= attempts; i += 1)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "$name is healthy: $url"
      return 0
    fi
    sleep "$sleep_seconds"
  done

  echo "$name did not become healthy: $url" >&2
  return 1
}

up() {
  local extra_args=()
  if [[ "${1:-}" == "--build" ]]; then
    extra_args+=(--build)
  elif [[ -n "${1:-}" ]]; then
    echo "Unsupported option for up: $1" >&2
    usage
    exit 1
  fi

  run_compose "$GROBID_DIR" up -d
  run_compose "$MINERU_DIR" up -d "${extra_args[@]}"
}

down() {
  run_compose "$MINERU_DIR" down
  run_compose "$GROBID_DIR" down
}

restart() {
  local extra_arg="${1:-}"
  down
  if [[ -n "$extra_arg" ]]; then
    up "$extra_arg"
  else
    up
  fi
}

ps_all() {
  echo "[grobid]"
  run_compose "$GROBID_DIR" ps
  echo
  echo "[mineru-api]"
  run_compose "$MINERU_DIR" ps
}

logs() {
  local target="${1:-all}"

  case "$target" in
    grobid)
      run_compose "$GROBID_DIR" logs -f grobid
      ;;
    mineru-api)
      run_compose "$MINERU_DIR" logs -f mineru-api
      ;;
    all)
      echo "Specify a service: grobid or mineru-api" >&2
      exit 1
      ;;
    *)
      echo "Unknown service: $target" >&2
      exit 1
      ;;
  esac
}

health() {
  wait_for_http "grobid" "http://127.0.0.1:8070/api/isalive"
  wait_for_http "mineru-api" "http://127.0.0.1:8000/health"
}

config() {
  run_compose "$GROBID_DIR" config >/dev/null
  run_compose "$MINERU_DIR" config >/dev/null
  echo "Docker Compose configuration is valid."
}

main() {
  require_cmd docker

  local command="${1:-}"
  shift || true

  case "$command" in
    up)
      up "$@"
      ;;
    down)
      down
      ;;
    restart)
      restart "$@"
      ;;
    ps)
      ps_all
      ;;
    logs)
      logs "$@"
      ;;
    health)
      require_cmd curl
      health
      ;;
    config)
      config
      ;;
    ""|-h|--help|help)
      usage
      ;;
    *)
      echo "Unknown command: $command" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
