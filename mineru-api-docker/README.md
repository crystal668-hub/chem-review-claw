# MinerU API Docker

This folder packages a local `mineru-api` service that listens on `127.0.0.1:8000` via Docker Compose.

## Quick start

```bash
sg docker -c 'cd /home/dministrator/.openclaw/workspace/mineru-api-docker && docker compose up -d --build'
```

For long-term startup/shutdown management, a user-level systemd unit is installed at:

- `~/.config/systemd/user/mineru-api-docker.service`

Useful commands:

```bash
systemctl --user daemon-reload
systemctl --user enable --now mineru-api-docker.service
systemctl --user restart mineru-api-docker.service
systemctl --user status mineru-api-docker.service
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
- Hugging Face / ModelScope caches are bind-mounted from `/home/dministrator/.cache/...` so the container can reuse models that were already downloaded on the host.
- A writable XDG cache stays in the named volume `mineru-xdg-cache`.
- The compose file also forwards the local proxy via `host.docker.internal:10090`; adjust or remove those env vars if your network setup changes.
- `gpus: all` is enabled because this machine already exposes an NVIDIA runtime through Docker.
- `openclaw-gateway.service` now has a user-level drop-in that orders it after `mineru-api-docker.service` so the fixed MinerU endpoint is available first.
- The current shell may still need `sg docker -c ...` until group membership refreshes in a new login session.
