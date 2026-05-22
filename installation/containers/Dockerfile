FROM nvidia/cuda:12.5.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Must run as root: DNAnexus swiss-army-knife invokes
# `docker run -v /home/dnanexus/out/out:/tmp -w /tmp` with no --user flag and
# no chown of the mount, so non-root UIDs get EACCES on output writes.
# dsub runs containers as root by default, so this is fine there too.
# Refs: https://academy.dnanexus.com/docker/swissarmyknife
#       https://community.ukbiobank.ac.uk/hc/en-gb/community/posts/16019643692061

COPY --chmod=755 entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
