FROM continuumio/miniconda3:25.1.1-2

ARG PYTHON_VERSION=3.11
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN conda create -y -n mineru python=${PYTHON_VERSION} pip \
    && conda clean -afy

SHELL ["conda", "run", "--no-capture-output", "-n", "mineru", "/bin/bash", "-lc"]

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    XDG_CACHE_HOME=/models/xdg-cache \
    NO_PROXY=127.0.0.1,localhost,::1 \
    no_proxy=127.0.0.1,localhost,::1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["conda", "run", "--no-capture-output", "-n", "mineru", "mineru-api", "--host", "0.0.0.0", "--port", "8000"]
