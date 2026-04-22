ARG BASE_IMAGE=continuumio/miniconda3:25.1.1-2
FROM ${BASE_IMAGE}

ARG PYTHON_VERSION=3.11
ARG BUILD_HTTP_PROXY=
ARG BUILD_HTTPS_PROXY=
ARG BUILD_NO_PROXY=127.0.0.1,localhost,::1,host.docker.internal
ARG PIP_INDEX_URL=
ARG PIP_EXTRA_INDEX_URL=
ARG PIP_TRUSTED_HOST=
WORKDIR /app

ENV HTTP_PROXY=${BUILD_HTTP_PROXY} \
    HTTPS_PROXY=${BUILD_HTTPS_PROXY} \
    NO_PROXY=${BUILD_NO_PROXY} \
    http_proxy=${BUILD_HTTP_PROXY} \
    https_proxy=${BUILD_HTTPS_PROXY} \
    no_proxy=${BUILD_NO_PROXY} \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        tini \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
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
    no_proxy=127.0.0.1,localhost,::1 \
    HTTP_PROXY= \
    HTTPS_PROXY= \
    http_proxy= \
    https_proxy= \
    PIP_INDEX_URL= \
    PIP_EXTRA_INDEX_URL= \
    PIP_TRUSTED_HOST=

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["mineru-api", "--host", "0.0.0.0", "--port", "8000"]
