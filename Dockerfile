FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/root/.local/bin:$PATH \
    UV_PYTHON=3.11 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && uv python install 3.11

WORKDIR /artifact
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --extra paper --no-install-project
COPY . .
RUN uv sync --frozen --extra paper

CMD ["uv", "run", "el-token", "paper", "verify"]
