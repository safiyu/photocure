FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip ca-certificates gosu curl \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/photocure/wheels /data

COPY packaging/deb/photocure-wrapper /usr/bin/photocure
COPY packaging/deb/bootstrap.py /opt/photocure/bootstrap.py
COPY packaging/docker/entrypoint.sh /entrypoint.sh

# Build the wheel from source; a failure must fail the image build.
COPY . /src
RUN cd /src && python3 -m pip wheel --no-deps -w /opt/photocure/wheels .

RUN chmod +x /usr/bin/photocure /entrypoint.sh /opt/photocure/bootstrap.py

VOLUME /data
EXPOSE 8080
# The first start installs the AI libraries (up to a few GB), so allow it plenty of time.
HEALTHCHECK --interval=30s --timeout=5s --start-period=1800s --retries=3 \
    CMD python3 -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/healthz', timeout=3)"

ENTRYPOINT ["/entrypoint.sh"]
