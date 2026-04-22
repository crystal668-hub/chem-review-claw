# GROBID Docker Service

This folder packages the local `grobid` service via Docker Compose.

## Current deployment shape

The current Compose service uses:

- Service name: `grobid`
- Container name: `chemqa-grobid`
- Image: `grobid/grobid:0.8.2.1-crf`
- Restart policy: `unless-stopped`
- Published port: `127.0.0.1:8070 -> 8070/tcp`
- OpenClaw env: `GROBID_URL=http://localhost:8070`

## Quick start

```bash
cd ~/.openclaw/workspace/grobid-docker
docker compose up -d
```

## Service management

For long-term startup/shutdown management on Linux, a user-level systemd unit can be installed at:

- `~/.config/systemd/user/grobid-docker.service`

Useful commands:

```bash
systemctl --user daemon-reload
systemctl --user enable --now grobid-docker.service
systemctl --user restart grobid-docker.service
systemctl --user status grobid-docker.service
```

## Quick checks

Health:

```bash
curl -fsS http://127.0.0.1:8070/api/isalive
curl -fsS http://127.0.0.1:8070/api/version
```

Container status:

```bash
cd ~/.openclaw/workspace/grobid-docker
docker compose ps
```

Real request smoke tests:

```bash
curl -X POST -F 'input=@/path/to/paper.pdf' http://127.0.0.1:8070/api/processHeaderDocument
curl -X POST -F 'input=@/path/to/paper.pdf' http://127.0.0.1:8070/api/processFulltextDocument
```

## Routine operations

Logs:

```bash
cd ~/.openclaw/workspace/grobid-docker
docker compose logs --tail 200 grobid
docker compose logs -f grobid
```

Restart:

```bash
cd ~/.openclaw/workspace/grobid-docker
docker compose restart grobid
```

Stop / start:

```bash
cd ~/.openclaw/workspace/grobid-docker
docker compose stop grobid
docker compose up -d grobid
```

## Recreate the service

If the container is deleted or needs a clean recreate, run:

```bash
cd ~/.openclaw/workspace/grobid-docker
docker compose down
docker compose up -d
```

Then verify:

```bash
curl -fsS http://127.0.0.1:8070/api/isalive
curl -fsS http://127.0.0.1:8070/api/version
```

## Notes

- The Compose file binds to `127.0.0.1:8070`, so the service is local-only by default.
- The service can be managed together with `mineru-api` from the repo root via `bash scripts/docker_services.sh ...`.
