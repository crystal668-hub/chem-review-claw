# MinerU API Docker

This folder packages a local `mineru-api` service that listens on `127.0.0.1:8000` via Docker Compose.

## Quick start

```bash
sg docker -c 'cd /home/dministrator/.openclaw/workspace/mineru-api-docker && docker compose up -d --build'
```

Check health:

```bash
sg docker -c 'curl -fsS http://127.0.0.1:8000/health'
sg docker -c 'cd /home/dministrator/.openclaw/workspace/mineru-api-docker && docker compose ps'
```

Logs:

```bash
sg docker -c 'cd /home/dministrator/.openclaw/workspace/mineru-api-docker && docker compose logs -f mineru-api'
```

Stop:

```bash
sg docker -c 'cd /home/dministrator/.openclaw/workspace/mineru-api-docker && docker compose down'
```

## Notes

- Current OpenClaw config can keep `MINERU_API_URL=http://127.0.0.1:8000`.
- The container binds only to loopback, so it is not exposed on the LAN.
- Model caches persist in the named volume `mineru-model-cache`.
- `gpus: all` is enabled because this machine already exposes an NVIDIA runtime through Docker.
- The current shell may still need `sg docker -c ...` until group membership refreshes in a new login session.
