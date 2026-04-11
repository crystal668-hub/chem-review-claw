# GROBID Docker Service

This folder documents the current local GROBID deployment.

## Current deployment shape

The active service is a standalone Docker container, not a Compose stack:

- Container name: `chemqa-grobid`
- Image: `grobid/grobid:0.8.2.1-crf`
- Restart policy: `unless-stopped`
- Published port: `0.0.0.0:8070 -> 8070/tcp`
- OpenClaw gateway env: `GROBID_URL=http://localhost:8070`

## Quick checks

Health:

```bash
curl -fsS http://127.0.0.1:8070/api/isalive
curl -fsS http://127.0.0.1:8070/api/version
```

Container status:

```bash
sg docker -c 'docker ps --filter name=chemqa-grobid'
sg docker -c 'docker inspect chemqa-grobid --format "{{.State.Status}} {{.HostConfig.RestartPolicy.Name}}"'
```

Real request smoke tests:

```bash
curl -X POST -F 'input=@/path/to/paper.pdf' http://127.0.0.1:8070/api/processHeaderDocument
curl -X POST -F 'input=@/path/to/paper.pdf' http://127.0.0.1:8070/api/processFulltextDocument
```

## Routine operations

Logs:

```bash
sg docker -c 'docker logs --tail 200 chemqa-grobid'
sg docker -c 'docker logs -f chemqa-grobid'
```

Restart:

```bash
sg docker -c 'docker restart chemqa-grobid'
```

Stop / start:

```bash
sg docker -c 'docker stop chemqa-grobid'
sg docker -c 'docker start chemqa-grobid'
```

## Recreate the container

If the container is deleted or needs a clean rebuild, recreate it with:

```bash
sg docker -c 'docker rm -f chemqa-grobid || true'
sg docker -c "docker run -d --name chemqa-grobid --restart unless-stopped -p 8070:8070 grobid/grobid:0.8.2.1-crf"
```

Then verify:

```bash
curl -fsS http://127.0.0.1:8070/api/isalive
curl -fsS http://127.0.0.1:8070/api/version
```

## Notes

- The current container publishes on `0.0.0.0:8070`, not loopback-only. If you want it local-only, recreate it with `-p 127.0.0.1:8070:8070`.
- The service has already been verified with real API calls against both `/api/processHeaderDocument` and `/api/processFulltextDocument`.
- The current shell may still need `sg docker -c ...` until group membership refreshes in a new login session.
- If you later want parity with the MinerU setup, the next step would be to wrap this container in a dedicated user-level systemd unit or move it into a small Compose stack.
