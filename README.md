## Docker services

This project expects two long-lived local HTTP services:

- `GROBID` at `http://127.0.0.1:8070`
- `MinerU API` at `http://127.0.0.1:8000`

They are referenced by the default environment variables in `~/.openclaw/.env`:

- `GROBID_URL=http://localhost:8070`
- `MINERU_API_URL=http://127.0.0.1:8000`

Use the repo helper script to manage both services together:

```bash
cd ~/.openclaw/workspace
bash scripts/docker_services.sh up --build
bash scripts/docker_services.sh ps
bash scripts/docker_services.sh health
```

Common operations:

```bash
bash scripts/docker_services.sh down
bash scripts/docker_services.sh restart
bash scripts/docker_services.sh logs grobid
bash scripts/docker_services.sh logs mineru-api
```

Service-specific Compose projects live in:

- `grobid-docker/compose.yaml`
- `mineru-api-docker/compose.yaml`

Notes:

- Both services bind to loopback only and are not exposed on the LAN.
- `mineru-api` reuses host caches via `${HOST_HF_CACHE}` and `${HOST_MODELSCOPE_CACHE}` when set, or falls back to `${HOME}/.cache/...`.
- If your host does not need the forwarded proxy, remove or override the proxy environment variables in `mineru-api-docker/compose.yaml`.
