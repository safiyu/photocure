# Photocure web dashboard. One image for every machine: on first start it detects the
# hardware (NVIDIA, AMD, Intel or CPU) and installs the matching AI libraries into /data.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUSERBASE=/data/pylibs

# Extra system packages, for example a GPU runtime. Space separated, empty by default.
ARG APT_PACKAGES=""
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 $APT_PACKAGES \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY photocure ./photocure
COPY docker/entrypoint.sh /usr/local/bin/pcur-entrypoint
RUN chmod +x /usr/local/bin/pcur-entrypoint

# Leave these empty for the normal auto-detecting image. Set them only to bake the AI
# libraries into the image for offline use, together with PCUR_AUTO_SETUP=0 at run time.
ARG EXTRAS=""
# Python wheel index for torch. Empty means the default PyPI wheel (CPU on Linux). Examples:
#   https://download.pytorch.org/whl/cu124   NVIDIA
#   https://download.pytorch.org/whl/rocm6.2 AMD
#   https://download.pytorch.org/whl/xpu     Intel
ARG TORCH_INDEX=""
RUN if [ -n "$TORCH_INDEX" ]; then pip install torch torchvision --index-url "$TORCH_INDEX"; fi \
    && if [ -n "$EXTRAS" ]; then pip install ".[$EXTRAS]"; else pip install .; fi

# Non-root user. HOME lives in the data volume so model weights are cached once.
RUN useradd --uid 1000 --create-home --home-dir /data curator \
    && mkdir -p /photos /output /data \
    && chown -R curator:curator /data /output
USER curator
ENV HOME=/data \
    PCUR_CACHE_DIR=/data/.cache \
    PCUR_BROWSE_ROOTS=/photos,/output \
    PCUR_LIBRARY_ROOT=/output/Organized
WORKDIR /data
VOLUME ["/data"]
EXPOSE 8080

# The first start downloads the AI libraries (up to a few GB), so allow it plenty of time.
HEALTHCHECK --interval=30s --timeout=5s --start-period=1800s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/healthz', timeout=3)"

ENTRYPOINT ["pcur-entrypoint"]
CMD ["auto"]
