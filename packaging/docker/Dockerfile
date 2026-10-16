FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip ca-certificates gosu curl \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/photocure/wheels /data

COPY packaging/deb/photocure-wrapper /usr/bin/photocure
COPY packaging/deb/bootstrap.py /opt/photocure/bootstrap.py
COPY packaging/docker/entrypoint.sh /entrypoint.sh

# If building from source, build the wheel
COPY . /src
RUN cd /src && python3 -m pip wheel --no-deps -w /opt/photocure/wheels . >/dev/null 2>&1 || true

RUN chmod +x /usr/bin/photocure /entrypoint.sh /opt/photocure/bootstrap.py

VOLUME /data
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD /usr/bin/photocure --version >/dev/null 2>&1 || exit 1

ENTRYPOINT ["/entrypoint.sh"]
