# MinerU API Docker

This folder packages a local `mineru-api` service that listens on `127.0.0.1:8000` via Docker Compose.

## Quick start

```bash
cd ~/.openclaw/workspace/mineru-api-docker
docker compose up -d --build
```

If Docker Hub is slow or blocked on the current machine, this Compose file defaults the base image to:

```bash
docker.m.daocloud.io/continuumio/miniconda3:25.1.1-2
```

You can override it, along with build-time proxies and Python package indexes, via environment variables before `docker compose up`:

```bash
export MINERU_BASE_IMAGE=continuumio/miniconda3:25.1.1-2
export MINERU_BUILD_HTTP_PROXY=http://host.docker.internal:10090
export MINERU_BUILD_HTTPS_PROXY=http://host.docker.internal:10090
export MINERU_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export MINERU_PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
docker compose up -d --build
```

For long-term startup/shutdown management on Linux, a user-level systemd unit can be installed at:

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
curl -fsS http://127.0.0.1:8000/health
cd ~/.openclaw/workspace/mineru-api-docker
docker compose ps
```

Logs:

```bash
cd ~/.openclaw/workspace/mineru-api-docker
docker compose logs -f mineru-api
```

Stop:

```bash
cd ~/.openclaw/workspace/mineru-api-docker
docker compose down
```

## Notes

- Current OpenClaw config can keep `MINERU_API_URL=http://127.0.0.1:8000`.
- The container binds only to loopback, so it is not exposed on the LAN.
- Hugging Face / ModelScope caches are bind-mounted from `${HOST_HF_CACHE}` and `${HOST_MODELSCOPE_CACHE}` when provided. Otherwise Docker resolves them from `${HOME}/.cache/...` on the current host.
- A writable XDG cache stays in the named volume `mineru-xdg-cache`.
- No build-time or runtime proxy is enabled by default. Set `MINERU_BUILD_HTTP_PROXY`, `MINERU_BUILD_HTTPS_PROXY`, `MINERU_RUNTIME_HTTP_PROXY`, or `MINERU_RUNTIME_HTTPS_PROXY` only when the current host actually exposes a reachable proxy.
- Build-time network knobs are configurable through `MINERU_BASE_IMAGE`, `MINERU_BUILD_HTTP_PROXY`, `MINERU_BUILD_HTTPS_PROXY`, `MINERU_PIP_INDEX_URL`, `MINERU_PIP_EXTRA_INDEX_URL`, and `MINERU_PIP_TRUSTED_HOST`.
- Runtime Hugging Face access defaults to `HF_ENDPOINT=https://hf-mirror.com` with public DNS resolvers `223.5.5.5` and `119.29.29.29` so the container can resolve and download MinerU models on networks where `huggingface.co` is not directly reachable.
- You can override those runtime network defaults with `MINERU_HF_ENDPOINT`, `MINERU_HF_HUB_DISABLE_XET`, `MINERU_DNS_1`, and `MINERU_DNS_2`.
- The compose file intentionally does not request GPUs so the same project layout can be restored on Apple Silicon Macs.
- Any OpenClaw service manager should start this container before launching workflows that assume a fixed `MINERU_API_URL`.
