FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv \
    git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace

# Install Python deps first so this layer stays cached when only code changes.
COPY installation/containers/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Bake the codebase in. AoU Batch VMs cannot reach github.com (no external IP,
# no NAT through the VPC-SC perimeter), so runtime clone is not viable for
# dsub jobs. Build-time COPY rides in via the GitHub Actions runner.
COPY . /workspace/Delphi
RUN pip install --no-cache-dir --no-deps -e /workspace/Delphi

# Must run as root: DNAnexus swiss-army-knife invokes
# `docker run -v /home/dnanexus/out/out:/tmp -w /tmp` with no --user flag and
# no chown of the mount, so non-root UIDs get EACCES on output writes.
# dsub runs containers as root by default, so this is fine there too.
# Refs: https://academy.dnanexus.com/docker/swissarmyknife
#       https://community.ukbiobank.ac.uk/hc/en-gb/community/posts/16019643692061

COPY --chmod=755 installation/containers/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
